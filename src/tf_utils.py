import random
import numpy as np
import tensorflow as tf


def dataset_to_batched_tensors(dataset, batch_size, tar_dim, inp_dim):
    batch_cutoffs = range(0, len(dataset), batch_size)
    num_batches = len(batch_cutoffs) - 1

    random.seed()
    random.shuffle(dataset)

    def generator():
        for i in range(num_batches):
            batch = dataset[batch_cutoffs[i]: batch_cutoffs[i + 1]]
            tar = [ex[0] for ex in batch]
            inp = [ex[1] for ex in batch]
            summaries = tf.keras.preprocessing.sequence.pad_sequences(tar, maxlen=tar_dim, padding='post',
                                                                      truncating='post', dtype='int32')
            codes = tf.keras.preprocessing.sequence.pad_sequences(inp, maxlen=inp_dim, padding='post',
                                                                  truncating='post', dtype='int32')
            summaries = tf.convert_to_tensor(summaries)
            codes = tf.convert_to_tensor(codes)
            yield summaries, codes

    return generator(), num_batches


def top_k_preds(pred_perps, allow_duplication):
    shape = tf.shape(pred_perps)
    batch_size = shape[0]
    beam_width = shape[1]
    sorted_tokens = tf.argsort(pred_perps, axis=-1)
    batch_of_best_perp_indices = []
    batch_of_best_perps = []
    for i in tf.range(0, limit=batch_size, delta=1):
        best_perps = []
        best_perp_indices = []
        taken_tokens = []
        for j in tf.range(0, limit=beam_width, delta=1):
            for k in tf.range(0, limit=beam_width, delta=1):
                token_idx = sorted_tokens[i][j][k]
                if allow_duplication or token_idx not in taken_tokens:
                    pred_perp = pred_perps[i][j][token_idx]
                    indices = [i, j, token_idx]
                    if len(best_perps) < beam_width:
                        best_perps.append(pred_perp)
                        best_perp_indices.append(indices)
                        taken_tokens.append(token_idx)
                    else:
                        worst_best = int(np.argmax(best_perps))
                        if pred_perp < best_perps[worst_best]:
                            best_perps[worst_best] = pred_perp
                            best_perp_indices[worst_best] = indices
                            taken_tokens.append(token_idx)
        vals, indices = tf.math.top_k(-tf.convert_to_tensor(best_perps), k=beam_width)
        best_perp_indices = tf.gather(best_perp_indices, indices)
        best_perps = -vals
        batch_of_best_perp_indices.append(best_perp_indices)
        batch_of_best_perps.append(best_perps)
    preds_tensor = tf.convert_to_tensor(batch_of_best_perp_indices)
    perps_tensor = tf.convert_to_tensor(batch_of_best_perps)
    return preds_tensor, perps_tensor


def update_beams(beam_preds, top_k_pred_indices, conditional_pred_perps, new_beam_states):
    batch_of_new_states = []
    batch_of_new_preds = []
    batch_of_new_perps = []
    for batch in top_k_pred_indices:
        new_states = []
        new_preds = []
        new_perps = []
        for indices in batch:
            x = indices[0]
            y = indices[1]
            z = indices[2]
            new_states.append(new_beam_states[x][y])
            new_preds.append(z)
            new_perps.append(conditional_pred_perps[x][y][z])
        batch_of_new_states.append(new_states)
        batch_of_new_preds.append(new_preds)
        batch_of_new_perps.append(new_perps)
    new_states_tensor = tf.convert_to_tensor(batch_of_new_states)
    new_perps_tensor = tf.convert_to_tensor(batch_of_new_perps)
    new_preds_tensor = tf.expand_dims(tf.convert_to_tensor(batch_of_new_preds), -1)
    combined_preds_tensor = tf.concat((beam_preds, new_preds_tensor), -1)
    return new_states_tensor, combined_preds_tensor, new_perps_tensor


def beam_search_decode_new(initial_states, single_bsd_step, start_token, end_token, beam_width=10, max_len=50,
                           batch_size=64):
    """
    Beam search decoder
    :param initial_states: shape (num_to_decode, arbitrary...)
    :param single_bsd_step: a function that takes in the current set of predictions,
    of shape (size_of_batch, beam_width, step), along with the current state, of shape
    (size_of_batch, beam_width, arbitrary...), and returns predictions for the next token, of shape
    (size_of_batch, beam_width, vocab_size), and the new state, of shape (size_of_batch, beam_width, arbitrary...).
    The state is not modified by beam_search_decode, and is passed along to the next call of single_bsd_step.
    It can be used for a recurrent cell's state, for example.
    :param start_token: the tokenizer's start token
    :param end_token: the tokenizer's end token
    :param beam_width: the number of beams to keep at each step
    :param max_len: the maximum length of the decoded sequence
    :param batch_size: the decoding will take place in batches of this size
    :return:
    """

    num_batches = tf.cast(tf.math.ceil(
        tf.cast(tf.shape(initial_states)[0], tf.float32) / tf.cast(batch_size, tf.float32)
    ), tf.int32)

    all_beam_preds = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

    for batch_num in tf.range(0, limit=num_batches, delta=1):

        size_of_batch = tf.minimum(batch_size, tf.shape(initial_states)[0] - batch_num * batch_size)
        initial_states_batch = initial_states[batch_num * batch_size: batch_num * batch_size + size_of_batch]

        beam_states = tf.repeat(tf.expand_dims(initial_states_batch, 1), beam_width, axis=1)
        beam_preds = tf.repeat(tf.expand_dims(tf.repeat(
            tf.expand_dims(tf.expand_dims(start_token, 0), 0), beam_width, axis=0
        ), 0), size_of_batch, axis=0)
        beam_perps = tf.repeat(tf.expand_dims(tf.repeat(tf.expand_dims(0.0, 0), beam_width, axis=0), 0),
                               size_of_batch, axis=0)

        allow_duplication = False
        for step in tf.range(0, limit=max_len, delta=1):

            all_end_tokens = tf.equal(beam_preds, end_token)
            already_finished_beams = tf.reduce_any(all_end_tokens, axis=-1)
            if tf.reduce_all(already_finished_beams):
                break

            pred_probs, new_beam_states = single_bsd_step(beam_preds, beam_states)

            vocab_size = tf.shape(pred_probs)[-1]
            expanded_old_perps = tf.repeat(tf.expand_dims(beam_perps, -1), vocab_size, axis=-1)
            finished_beams_broadcast = tf.repeat(tf.expand_dims(already_finished_beams, -1), vocab_size, axis=-1)
            conditional_pred_perps = tf.where(finished_beams_broadcast, x=expanded_old_perps,
                                              y=expanded_old_perps - tf.math.log(pred_probs))
            top_k_pred_indices, top_k_pred_perps = top_k_preds(conditional_pred_perps, allow_duplication)
            allow_duplication = True

            beam_states, beam_preds, beam_perps = update_beams(beam_preds, top_k_pred_indices, conditional_pred_perps,
                                                               new_beam_states)

        best_beam_preds = beam_preds[:, 0, :]
        all_beam_preds = all_beam_preds.write(batch_num, best_beam_preds)

    all_beam_preds_final = all_beam_preds.concat()
    return all_beam_preds_final


def beam_search_decode(initial_state, single_bsd_step, start_token, end_token, beam_width=10, max_len=50):
    final, initial_state = single_bsd_step(tf.expand_dims(start_token, 0), initial_state)
    predictions = tf.argsort(final, axis=-1, direction='DESCENDING').numpy()[0:beam_width]
    beams = []
    for k in range(beam_width):
        formed_candidate = ([start_token, predictions[k]],
                            -tf.math.log(final[predictions[k]]),
                            initial_state)
        beams.append(formed_candidate)
    for j in range(max_len - 1):
        candidates = []
        for k in range(beam_width):
            if beams[k][0][-1] == end_token:
                if len(candidates) < beam_width:
                    candidates.append(beams[k])
                else:
                    for m in range(len(candidates)):
                        if candidates[m][1] > beams[k][1]:
                            candidates[m] = beams[k]
                            break
            else:
                final, new_state = single_bsd_step(beams[k][0], beams[k][2])
                predictions = tf.argsort(final, axis=-1, direction='DESCENDING').numpy()[0:beam_width]
                for prediction in predictions:
                    formed_candidate = (beams[k][0] + [prediction],
                                        beams[k][1] + -tf.math.log(final[prediction]),
                                        new_state)
                    if len(candidates) < beam_width:
                        candidates.append(formed_candidate)
                    else:
                        for m in range(len(candidates)):
                            if candidates[m][1] > formed_candidate[1]:
                                candidates[m] = formed_candidate
                                break
        beams = candidates
        if all(beams[k][0][-1] == end_token for k in range(beam_width)):
            break
    lowest_perplexity_beam = beams[0]
    for k in range(1, beam_width):
        if beams[k][1] < lowest_perplexity_beam[1]:
            lowest_perplexity_beam = beams[k]
    return lowest_perplexity_beam
