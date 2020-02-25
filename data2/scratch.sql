
select question.Title, answer.Body
  from posts as question, posts as answer
  where question.AcceptedAnswerID = answer.Id and question.Tags like '%<c#>%'
        and answer.Body like '%<code>%</code>%' and  answer.Body not like '%</code>%<code>%'
        and LEN(answer.Body) > 30 and LEN(answer.Body) < 500