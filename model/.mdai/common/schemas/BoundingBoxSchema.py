from marshmallow import Schema, fields


class BoundingBoxSchema(Schema):
    annotationText = fields.String(description="Annotation Text")
    x1 = fields.Number()
    y1 = fields.Number()
    length = fields.Number()
    breadth = fields.Number()
