# simple-aws

This is a wrapper over AWS boto3 to simplify many common AWS tasks.

## DynamoDb

ddb.SingleTable implements the [single table]() approach. Simply provide a mapping from your [Pydantic]() model fields to the table keys, and you can create use case specific queries...