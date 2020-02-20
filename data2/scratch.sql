select TOP 200 Id, AcceptedAnswerId, Body, Tags, Id AS [Post Link]
  from posts where PostTypeId=1 and Tags like '%<c#>%' and AcceptedAnswerID IS NOT NULL