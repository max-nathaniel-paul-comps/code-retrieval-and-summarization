import tensorflow as tf


def tf_dataset(dataset, batch_size, max_tar_len, max_inp_len, tar_tokenizer, inp_tokenizer,
               shuffle_buffer_size=20000):

    def tf_tokenize(tar, inp):
        def tokenize(tar_py, inp_py):
            tokenized_tar = tar_tokenizer.tokenize_text(tar_py.numpy().decode('utf-8'))
            tokenized_inp = inp_tokenizer.tokenize_text(inp_py.numpy().decode('utf-8'))
            return tokenized_tar, tokenized_inp

        tar, inp = tf.py_function(tokenize, [tar, inp], [tf.int64, tf.int64])
        tar.set_shape([None])
        inp.set_shape([None])

        return tar, inp

    def filter_max_length(tar, inp):
        return tf.logical_and(tf.size(tar) <= max_tar_len,
                              tf.size(inp) <= max_inp_len)

    dataset = tf.data.Dataset.from_generator(dataset, (tf.string, tf.string))
    tokenized = dataset.map(tf_tokenize)
    filtered = tokenized.filter(filter_max_length)
    shuffled = filtered.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
    batched = shuffled.padded_batch(batch_size, ([max_tar_len], [max_inp_len]),
                                    drop_remainder=True).prefetch(1)
    return batched


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
