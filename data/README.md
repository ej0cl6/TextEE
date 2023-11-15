## Processed Data Format

The data file should contain a json string in each line. The json format is as follows.

```JSON5
{
  "doc_id": "DOC101",  // required, an unique string for each document
  "wnd_id": "DOC101-2",  // required, an unique string for each text instance in the document
  "text": "James will start his first job at Google in June.",  // required, the plain text of the instance
  "lang": "en",  // required, the langauge for this instance
  "tokens": ["James", "will", "start", "his", "first", "job", "at", "Google", "in", "June", "."],  // required, the list of tokens in the sentence  
  "entity_mentions": [  // required, the list of entity mentions
    {
      "id": "DOC101-2-E0",
      "entity_type": "PER", 
      "text": "James",
      "start": 0,
      "end": 1, 
    }, 
    {
      "id": "DOC101-2-E1",
      "entity_type": "ORG", 
      "text": "Google",
      "start": 7,
      "end": 8,
    },
    ...
  ],  
  "event_mentions": [  // required, the list event mentions
    {
      "id": "DOC101-2-EV0", 
      "event_type": "START_JOB", 
      "trigger": {  // trigger span
        "text": "start",
        "start": 2,
        "end": 3,
      },
      "arguments": [  // the list of argument-role pairs
        {
          "entity_id": "DOC101-2-E1",  // corresonding to entity mentions
          "role": "company",
          "text": "Google",
          "start": 7,
          "end": 8,
        },
        {
          "entity_id": "DOC101-2-E3",  // corresonding to entity mentions
          "role": "time",
          "text": "June",
          "start": 9,
          "end": 10,
        },
        ...
      ],
    },
    {
    ...
    }
  ],
  "relation_mentions": [  // optional, the list of relation mentions
    {
      "id": "DOC101-2-R1",
      "relation_type": "employed_by",
      "arguments": [
        {
          "entity_id": "DOC101-2-E0",  // corresonding to entity mentions
          "role": "Arg-1",
          "text": "James",
          "start": 0,
          "end": 1,
        },
        {
          "entity_id": "DOC101-2-E1",  // corresonding to entity mentions
          "role": "Arg-2",
          "text": "Google",
          "start": 7,
          "end": 8,
        },
      ],
    },
    ...
  ], 
}
```


