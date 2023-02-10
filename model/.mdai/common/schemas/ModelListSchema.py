from marshmallow import Schema, fields

class ModelDetails(Schema):
    model = fields.String(description="Models")
    version = fields.String(description="Model Version")


class ModelListSchema(Schema):
    status = fields.String(default='Success', required=True)
    data = fields.List(fields.Nested(ModelDetails))