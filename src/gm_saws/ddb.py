from __future__ import annotations
from enum import Enum
from typing import Dict, List, Any
from pydantic import BaseModel as PydanticBaseModel
from boto3.dynamodb.types import STRING, NUMBER, BINARY, TypeSerializer, TypeDeserializer
import boto3
from boto3.dynamodb.conditions import Key
import pickle
from decimal import Decimal
from numbers import Number
import sys
import time


class IndexType(Enum):
    MAIN_TABLE = 0
    GLOBAL_SECONDARY = 1


class KeyType(str, Enum):
    PARTITION_KEY = 'pk'
    SORT_KEY = 'sk'

    def getDynamoDbKeyType(self):
        if self == KeyType.PARTITION_KEY:
            return 'HASH'
        if self == KeyType.SORT_KEY:
            return 'RANGE'


class QueryConditionType(Enum):
    EQ = 0
    LE = 1
    LT = 2
    GE = 3
    GT = 4
    BEGINS_WITH = 5
    BETWEEN = 6


class QueryTerm:
    """
    match_condition must be one of the codes defined for ComparisonOperator in https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/LegacyConditionalParameters.KeyConditions.html
    """

    def __init__(self, value: str, type: QueryConditionType) -> None:
        self.match_value: str = value
        self.match_type: QueryConditionType = type


class boto3_wrap:
    """Helper class-methods to perform common tasks using boto3"""

    @classmethod
    def getDynamoDbResource(cls, kwargs = None, local_db: bool = False):
        """Returns a boto3 resource for DynamoDb service"""

        if kwargs is None:
            if local_db:
                # TODO: Consider getting a config file path and read these local_db args from there
                return boto3.resource(service_name = 'dynamodb', endpoint_url = 'http://localhost:8000',
                                      aws_access_key_id = 'fakekeyid', aws_secret_access_key = 'fakeaccesskey')
            else:
                return boto3.resource(service_name = 'dynamodb', region_name = 'us-east-1')
        else:
            return boto3.resource(service_name = 'dynamodb', **kwargs)

    @classmethod
    def checkTable(cls, dyn_resource, table_name: str):
        """Checks if table exists"""
        tables = dyn_resource.meta.client.list_tables()
        return (table_name in tables['TableNames'])
    
    @classmethod
    def getTableColumns(cls, dyn_resource, table_name: str):
        if cls.checkTable(dyn_resource, table_name):
            return dyn_resource.meta.client.describe_table(TableName=table_name)['Table']['AttributeDefinitions']
        else:
            return None
        
    @classmethod
    def deleteTable(cls, dyn_resource, table_name: str, wait=True):
        client = dyn_resource.meta.client
        client.delete_table(TableName = table_name)

        # Wait for complete table deletion
        if wait:
            waiter = client.get_waiter('table_not_exists')
            waiter.wait(TableName = table_name)
        

class SingleTable:
    """A wrapper over boto3 API to map a group of related Pydantic Models to a single dynamodb table"""

    # Name for main table index
    PRIMARY = 'primary'

    def __init__(self, dyn_resource, table_name: str, rcu: int = 1, wcu: int = 1) -> None:
        # boto3 ddb resource
        self.dyn_resource = dyn_resource

        # Table name
        self.table_name = table_name

        # DynamoDb read/write capacity units
        self.ddb_capacity: Dict[str: Dict[str, int]] = {self.PRIMARY: {'rcu': rcu, 'wcu': wcu}}

        # Indices and Keys
        self.indices_keys: Dict[str, Dict[KeyType, tuple]] = {self.PRIMARY: {KeyType.PARTITION_KEY: None, KeyType.SORT_KEY: None}}

        # Entity Map
        self.entity_map: Dict[str, Dict[str, tuple(str, List[str])]] = {}

        # Check if exists, load Table
        if boto3_wrap.checkTable(self.dyn_resource, self.table_name):
            self.table = self.dyn_resource.Table(self.table_name)
        else: self.table = None
        
        # To lock further schema update once table created
        self.schema_locked = False

        # Entity.attribute to table.key map
        self.reverse_map = {}

        # Boto3 serializer and deserializer
        self.serializer = TypeSerializer()
        self.deserializer = TypeDeserializer()


    def add_key(self, index: str, indexType: IndexType, key: str, type: KeyType, dataType: str = STRING, rcu: int = 1, wcu: int = 1):
        """Define key(type) for index. The specified key must be one of table attributes"""

        # TODO: how to store indexType

        if self.schema_locked:
            raise Exception('Schema locked for editing, recreate table with new schema')
        
                # DynamoDb read/write capacity units
        self.ddb_capacity[index] = {'rcu': rcu, 'wcu': wcu}


        if index is None:
            index = self.PRIMARY

        if index not in self.indices_keys:
            # Init dict
            self.indices_keys[index] = {KeyType.PARTITION_KEY: None, KeyType.SORT_KEY: None}

        # Update specified key, and assume type=STRING
        self.indices_keys[index][type] = (key, dataType)


    def map_entity(self, entity: PydanticBaseModel, key_attribs_map: Dict[str, tuple]):
        """
        Maps entity to db table fields
        key_attribs_map maps table field to a tuple(expr, list[entity attributes])
        """

        if self.schema_locked:
            raise Exception('Schema locked for editing, recreate table with new schema')
        
        # Get Class Name
        cls_name = entity.__name__

        # Get type hints for PydanticBaseModel
        # fields_info = entity.__fields__

        # Check if mapping exists
        if cls_name not in self.entity_map:
            self.entity_map[cls_name] = {}

        # Add mappings
        for key, value in key_attribs_map.items():
            # Add mapping
            self.entity_map[cls_name][key] = value

            # Update field type, if needed
            # expr = value[0]
            # if expr is None or expr == '':
            #     # Get 1st attribute's type
            #     attrib = value[1][0]
            #     t = fields_info[attrib].annotation

            #     # Select dyn field type
            #     # TODO: other type checks?
            #     if issubclass(t, int):
            #         self.table_field_type_overrides[key] = NUMBER
            #     if issubclass(t, bytes):
            #         self.table_field_type_overrides[key] = BINARY
        
    def get_entity_map(self, entity: PydanticBaseModel):
        return self.entity_map[entity.__name__]
    
    def createTable(self, forceCreate=False):
        keySchema = []
        attributeDefinitions = []
        globalSecondaryIndexes = []

        for idx in self.indices_keys:
            gsi = {'IndexName': idx, 'KeySchema': [], 'Projection': {'ProjectionType': "ALL"},
                   'ProvisionedThroughput': {'ReadCapacityUnits': self.ddb_capacity[idx]['rcu'], 'WriteCapacityUnits': self.ddb_capacity[idx]['wcu']}}
            for k, v in self.indices_keys[idx].items():
                key = {'AttributeName': v[0], 'KeyType': k.getDynamoDbKeyType()}
                attrib = {'AttributeName': v[0], 'AttributeType': v[1]}

                # If not already added
                if attrib not in attributeDefinitions:
                    attributeDefinitions.append(attrib)

                if idx == self.PRIMARY:
                    keySchema.append(key)
                else:
                    gsi['KeySchema'].append(key)
            
            if idx != self.PRIMARY:
                globalSecondaryIndexes.append(gsi)
        
        table_exists = boto3_wrap.checkTable(self.dyn_resource, self.table_name)

        if table_exists:
            if forceCreate:
                boto3_wrap.deleteTable(self.dyn_resource, self.table_name)
            else:
                print('Table already exists, delete if need to change schema')
                return

        # Create table
        print('Creating DynamoDb Table...')
        self.table = self.dyn_resource.create_table(
            TableName=self.table_name,
            KeySchema=keySchema,
            AttributeDefinitions=attributeDefinitions,
            ProvisionedThroughput={'ReadCapacityUnits': self.ddb_capacity[self.PRIMARY]['rcu'], 'WriteCapacityUnits': self.ddb_capacity[self.PRIMARY]['wcu']},
            GlobalSecondaryIndexes=globalSecondaryIndexes
        )

        self.table.wait_until_exists()

        # Reverse map
        self.gen_reverse_map()


    def gen_reverse_map(self):
        for cls_name, attrib_map in self.entity_map.items():
            for tbl_key, tpl in attrib_map.items():
                for attrib in tpl[1]:
                    revmap_key = (cls_name, attrib)
                    if revmap_key not in self.reverse_map:
                        self.reverse_map[revmap_key] = [tbl_key]
                    else:
                        self.reverse_map[revmap_key].append(tbl_key)

        # Lock schema
        self.schema_locked = True

    @classmethod
    def _get_db_val_from_raw(cls, raw_val):
        # If enum, store value (expected to be either int or str)
        if isinstance(raw_val, Enum):
            return raw_val.value
        elif isinstance(raw_val, Number):
            # REF: https://github.com/boto/boto3/issues/665#issuecomment-340260257
            return Decimal(str(raw_val))
        else:
            return raw_val
        # TODO: datetime, float?


    @classmethod
    def _get_db_val(cls, entity, attrib):
        """Converts Pydantic attribute value to suitable type to store in db"""

        # Get attribute value, convert to dynamodb storable type if necessary
        raw_val = entity.__getattribute__(attrib)
        return cls._get_db_val_from_raw(raw_val)


    @classmethod
    def _get_table_field_value(cls, item, cls_name, expr: str, attrib_list):
        if expr is None or expr == '':
            # Return original attribute value if no expr specified
            return cls._get_db_val(item, attrib_list[0])
        else:
            # Build a list of args converted to str
            str_val_list = []
            for attrib in attrib_list:
                raw_val = cls._get_db_val(item, attrib)
                str_val = str(raw_val)

                str_val_list.append(str_val)
        
        # Format as per expr and return
        val = expr.format(*str_val_list).replace('@ClassName', cls_name)
        return val

    def _float_to_decimal(self, v):
        return Decimal(v) if isinstance(v, float) else v
        
    def item_2_ddb(self, item: type):
        # Pickle the item get ddb compatible type
        pickled = pickle.dumps(item)
        # Now serialize to store in ddb
        ddb_item = self.serializer.serialize(pickled)
        # boto3_serialize = False
        # if boto3_serialize:
        #     item_dict = item.model_dump()
        #     ddb_item = {}
        #     for k, v in item_dict.items():
        #         ddb_item[k] = serializer.serialize(self._float_to_decimal(v))
        return ddb_item
    
    def ddb_2_item(self, ddb_item, entity: PydanticBaseModel = None):
        try:
            # Somehow ddb item is returned as Binary, convert to bytes
            if isinstance(ddb_item, dict):
                ddb_item = {'B': bytes(ddb_item['B'])}
            else:
                # only the binary item, convert to dict
                ddb_item = {'B': bytes(ddb_item)}
            # De-serialize to pickled, and then un-pickle
            pickled = bytes(self.deserializer.deserialize(ddb_item))
            # Un-pickle
            item = pickle.loads(pickled)

            # TODO: If entity specified, check if item has all fields required in entity, else fill with defaults - this is useful when newer fields are added since item was stored in db
            # validated_item = entity.model_validate(item, )

            return item
        except Exception as e:
            # TODO: log error
            return None

    # REF: https://gist.githubusercontent.com/bosswissam/a369b7a31d9dcab46b4a034be7d263b2/raw/f99d210019c1fac6bb46d2da81dcdf5ef9932172/pysize.py
    def _get_size(self, obj, seen=None):
        """Recursively finds size of objects"""
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        # Important mark as seen *before* entering recursion to gracefully handle
        # self-referential objects
        seen.add(obj_id)
        if isinstance(obj, dict):
            size += sum([self._get_size(v, seen) for v in obj.values()])
            size += sum([self._get_size(k, seen) for k in obj.keys()])
        elif hasattr(obj, '__dict__'):
            size += self._get_size(obj.__dict__, seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([self._get_size(i, seen) for i in obj])
        return size

    def sleep_to_manage_wcu(self, item_dict: dict, delay: int = None, sleep: bool=True):
        item_size = self._get_size(item_dict)
        auto_dly = int(item_size / self.ddb_capacity[self.PRIMARY]['wcu'])
        if sleep:
            if delay is None:
                # Auto-add delay based on size of item - larger the item, larger the delay between writes
                # TODO: which WCU to use, min of all affected indices?
                delay = auto_dly

            if delay > 0:
                print(f'Written item of {item_size} bytes, now sleeping for {delay} ms...')
                time.sleep(delay / 1000)
        else:
            # Doesn't sleep but suggests caller to wait for auto_dly before writing into the table again
            return auto_dly

    def put_raw(self, key_map: dict, item: PydanticBaseModel, delay: int = None, sleep: bool=True):
        """Adds a table row based on specified key map. Pickles the item before saving and adds it to attribute=data.
        If delay is specified, sleeps for delay milli-seconds before returning, and this can be used to control write-capacity usage"""

        # TODO: (deep, to avoid key_map from getting modified) Copy the key-map and add pickled data
        table_row = key_map.copy()

        table_row['data'] = self.item_2_ddb(item)

        # Write to table
        # item_size = self._get_size(table_row['data'])
        # print(type(item), key_map)

        print('Writing item into database...')
        st = time.perf_counter()

        response = self.table.put_item(Item=table_row)
        
        print('Time taken to write item: {:6.3f} seconds'.format(time.perf_counter() - st))
        # print(response)
        # print()

        self.sleep_to_manage_wcu(table_row, delay=delay, sleep=sleep)
        # auto_dly = int(item_size / self.ddb_capacity[self.PRIMARY]['wcu'])
        # if sleep:
        #     if delay is None:
        #         # Auto-add delay based on size of item - larger the item, larger the delay between writes
        #         # TODO: which WCU to use, min of all affected indices?
        #         delay = auto_dly

        #     if delay > 0:
        #         print(f'Written {type(item)} of {item_size} bytes, now sleeping for {delay} ms...')
        #         time.sleep(delay / 1000)
        # else:
        #     # Doesn't sleep but suggests caller to wait for auto_dly before writing into the table again
        #     return auto_dly


    def get_raw(self, key_map: dict):
        """Gets an item (and un-pickles) from main table index using specified primary key (key_map can contain pk and sk, depending on how table was defined)"""

        response = self.table.get_item(Key=key_map)

        # expr = None
        # for k, v in key_map.items():
        #     if expr is None:
        #         expr = Key(k).eq(v)
        #     else:
        #         expr = expr & Key(k).eq(v)

        # response = self.table.query(KeyConditionExpression=expr)

        if response['ResponseMetadata']['HTTPStatusCode'] == 200 and 'Item' in response:
            ddb_item = response['Item']['data']
            item = self.ddb_2_item(ddb_item)
            return item

        # Not found
        return None


    def _getKeyValueItem_keyMap(self, key: str):
        # Stores {'pk': 'key-value-pair', 'sk: key, 'data': value} in primary table
        pk = self.indices_keys[self.PRIMARY][KeyType.PARTITION_KEY][0]
        sk = self.indices_keys[self.PRIMARY][KeyType.SORT_KEY][0]
        key_map = {pk: 'key-value-pair', sk: key}
        return key_map

    def putKeyValuePair(self, key: str, value: Any):
        self.put_raw(self._getKeyValueItem_keyMap(key), value)

    def getKeyValuePair(self, key: str) -> Any:
        return self.get_raw(self._getKeyValueItem_keyMap(key))
        

    def _get_primary_keys(self):
        # Extract primary table key names
        return [t[0] for t in self.indices_keys[self.PRIMARY].values()]

    def _is_primary_key(self, key: str):
        prim_keys = self._get_primary_keys()
        return (key in prim_keys)

    def _item_2_table_row(self, item: type, primary_key_only: bool = False):
        cls_name = item.__class__.__name__
        if cls_name in self.entity_map:
            table_row = {}
            for key, value in self.entity_map[cls_name].items():
                if primary_key_only and not self._is_primary_key(key):
                    continue

                # If expr is blank, attrib_list[0] is mapped to table 'key', otherwise, expr.format(**attrib_list)
                expr = value[0]
                # attrib_list can contain @ClassName in place of attribute name, e.g. [attrib1, attrib2, @ClassName, attrib3, ...]
                attrib_list = value[1]

                table_row[key] = self._get_table_field_value(item, cls_name, expr, attrib_list)
            
            return table_row
        else:
            return None


    def put_item(self, item: type, delay: int = None):
        """Add item into table based on entity-map specified during initialization.
        If delay is specified, sleeps for delay milli-seconds before returning, and this can be used to control write-capacity usage"""

        if self.table is None:
            raise Exception('Table not initialized')
        
        if not self.schema_locked:
            raise Exception('Schema not locked')
        
        # This is now done by put_raw()
        # # Pickle the data and store
        # table_row = {'data': pickle.dumps(item)}
        table_row = self._item_2_table_row(item)

        if table_row is not None:
            # Add item to table            
            self.put_raw(key_map=table_row, item=item, delay=delay)
        else:
            raise ValueError(f'Map {item.__class__.__name__} table before adding items')


    @classmethod
    def _build_QueryExpr(cls, cls_name, tbl_key, expr: str, constraints: Dict[str, QueryTerm], attribs):
        if len(attribs) <= 1:
            # Simple case, a single attrib
            qTerm = constraints[attribs[0]]

            # Requires Python >= 3.10
            match(qTerm.match_type):
                case QueryConditionType.EQ:
                    # Create db_val (replicated logic but no entity here)
                    db_val = cls._get_db_val_from_raw(qTerm.match_value)
                    if expr is None or expr == '':
                        return Key(tbl_key).eq(db_val)
                    else:
                        val = expr.format(db_val).replace('@ClassName', cls_name)
                        return Key(tbl_key).eq(val)
                case QueryConditionType.BETWEEN:
                    min_ = cls._get_db_val_from_raw(qTerm.match_value[0])
                    max_ = cls._get_db_val_from_raw(qTerm.match_value[1])
                    if expr is None or expr == '':
                        return Key(tbl_key).between(min_, max_)
                    else:
                        val_min = expr.format(min_).replace('@ClassName', cls_name)
                        val_max = expr.format(max_).replace('@ClassName', cls_name)
                        return Key(tbl_key).between(val_min, val_max)
                case _:
                    # No match
                    return True
        else:
            # Case when attribs count same as expr vars AND query type = EQ
            arg_list = []
            for attrib in attribs:
                qTerm = constraints[attrib]
                if qTerm.match_type == QueryConditionType.EQ:
                    arg_list.append(cls._get_db_val_from_raw(qTerm.match_value))

            # Try to evaluate expression if every attrib was added to arg_list
            if len(attribs) == len(arg_list):
                eval_match_val = expr.format(*arg_list).replace('@ClassName', cls_name)
                return Key(tbl_key).eq(eval_match_val)
            
            # Check if query type is same for each attrib
            # Get list of constrained value to be used with expr

            # cval_list = [constraints[attrib].match_value for attrib in attribs]
            # print(f'WIP: {tbl_key}, {expr}, {constraints}, {attribs}')
            # for attrib in attribs:
            #     qTerm = constraints[attrib]
            raise NotImplementedError('Queries with complex expression containing multiple attributes not supported yet.')
            

    def query(self, entity: PydanticBaseModel, constraints: Dict[str, QueryTerm], limit: int = None, get_raw_response: bool = False):
        """
        Queries the db to find entity_class instances based on conditions specified

        Parameters:
        constraints (dict): key=attribute name, value=QueryTerm
        """
        
        # Use Global Secondary Index to create different queries: https://pynamodb.readthedocs.io/en/stable/indexes.html
        # https://erudika.com/blog/2016/11/21/Saving-money-on-DynamoDB-with-Global-Secondary-Indexes/#:~:text=The%20shared%20table%20with%20a,s%20and%201%20write%2Fs.
        # https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/GSI.html

        # Get entity class name
        cls_name = entity.__name__

        # Check if entity has mapping to table
        if cls_name not in self.entity_map:
            raise KeyError('{0} has no mapping to database table, query cannot be created')
        
        # Multiple expressions are possible, store them in dict before executing the query
        query_expressions = {}

        constrained_attribs = set(constraints.keys())

        keyCondExpr = None

        # Iterate through table columns and check if a mapping attributes is constrained?
        for tbl_col, value in self.entity_map[cls_name].items():
            expr: str = value[0]
            attribs = value[1]

            # Check if attrib_list fully specified by constraints or not, get intersection
            intersect_ = set(attribs).intersection(constrained_attribs)
            if len(intersect_) == 0:
                # No constraint specified for this column
                continue

            if set(attribs).issubset(constrained_attribs):
                # tbl_col fully specified
                keyExpr = self._build_QueryExpr(cls_name, tbl_col, expr, constraints, attribs)
                query_expressions[tbl_col] = keyExpr
            else:
                # Try to build a starts-with expr - check if intersecting attributes match initial sub-list of attribs
                len_intersect = len(intersect_)
                sub_attrib_list = attribs[:len_intersect]
                if set(sub_attrib_list) == intersect_:
                    # Evaluate sub-expr
                    splittor = f'{len_intersect}'
                    sub_expr: str = expr.split('{' + splittor + '}')[0]

                    # Check that specified query for intersect_ attributes is EQ, and during the check, build arg-list
                    arg_list = []
                    for attr in sub_attrib_list:
                        qt = constraints[attr]
                        if qt.match_type == QueryConditionType.EQ:
                            arg_list.append(self._get_db_val_from_raw(qt.match_value))
                        else:
                            raise NotImplementedError(f'Partially specified query supported only for EQ type constraints')

                    eval_match_val = sub_expr.format(*arg_list).replace('@ClassName', cls_name)
                    keyExpr = Key(tbl_col).begins_with(eval_match_val)
                    query_expressions[tbl_col] = keyExpr
                    # print(eval_match_val)
                else:
                    # Find which attributes not constrained for this tbl_col
                    unconstrained = set(attribs).difference(constrained_attribs)
                    raise NotImplementedError(f'Partially specified queries not supported yet, {unconstrained} attributes not specified, and also unable to form a begins-with query since constrained attributes do not match a sublist of entity-map')
            
            if keyCondExpr is None:
                keyCondExpr = keyExpr
            else:
                keyCondExpr = keyCondExpr & keyExpr
        
        # Get the index to query on
        for idx, keyInfo in self.indices_keys.items():
            keys = [vals[0] for vals in keyInfo.values()]
            # Check if keys match
            if set(query_expressions.keys()).issubset(set(keys)):
                # print(idx, query_expressions)

                # Perform the query
                # TODO: add limit
                if idx != self.PRIMARY:
                    if limit is None:
                        response = self.table.query(IndexName=idx, KeyConditionExpression=keyCondExpr)
                    else:
                        response = self.table.query(IndexName=idx, KeyConditionExpression=keyCondExpr, Limit=limit)
                else:
                    if limit is None:
                        response = self.table.query(KeyConditionExpression=keyCondExpr)
                    else:
                        response = self.table.query(KeyConditionExpression=keyCondExpr, Limit=limit)

                # print(response['ResponseMetadata'])

                if get_raw_response:
                    return response
                else:
                    # Convert to entity list and return
                    return self.process_query_response(response, entity)
                
    def process_query_response(self, response, entity: PydanticBaseModel = None):
        result = []
        if response['ResponseMetadata']['HTTPStatusCode'] == 200 and 'Items' in response:
            for ddb_item in response['Items']:
                item = self.ddb_2_item(ddb_item['data'], entity)
                # Add if not None
                if item is not None:
                    result.append(item)

        return result
    
    def delete_item(self, item: type):
        item_dict = self._item_2_table_row(item, primary_key_only=True)
        self.delete_raw(item_dict)

    def delete_raw(self, item_dict: dict) -> bool:
        # Validate correctly specified prim-keys
        prim_keys = set(self._get_primary_keys())
        if (prim_keys.issubset(item_dict.keys())):
            prim_key_dict = {k:item_dict[k] for k in prim_keys}
            self.table.delete_item(Key=prim_key_dict)
        else:
            return False