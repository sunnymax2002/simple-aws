from gm_saws.ddb import SingleTable, boto3_wrap

engine = SingleTable(dyn_resource=boto3_wrap.getDynamoDbResource(), table_name="test_single_table")

print(engine)