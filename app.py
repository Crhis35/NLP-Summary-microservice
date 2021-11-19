#!/usr/bin/env python

from flask import Flask, request
from flask_restx import Api, Resource, fields
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS, cross_origin
from docsumfuncs import summarize

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='1.0', title='Text Summary API',
          description='NLP API to proceed text',
          license='MIT',
          )

ns = api.namespace('Summary', description='Summary operations')
summaryInput = api.model('SummaryInput', {
    'text': fields.String(required=True, description='Text to summarize'),
})

summaryModel = api.model('SummaryOutput', {
    'summary': fields.String(description='The summary of the text'),
    'fullText': fields.String(description='The summary with full text')
})

summaryOutput = api.model('Summary', {
    'status': fields.Integer(description='Status code'),
    'data': fields.Nested(
        summaryModel)
})

# parser = api.parser()
# parser.add_argument(
#     "text", type=str, required=True, help="The task details", location="form"
# )


@ns.route('/')
@api.route('/')
class Summary(Resource):

    @ns.doc('create_summary')
    @ns.expect(summaryInput)
    # @api.doc(parser=parser)
    @ns.marshal_with(summaryOutput)
    def post(self):
        """
            Creates a new summary

            """
        data = request.get_json()
        text = data['text']
        if(text):
            post = summarize(text)
            # You could also store a version of the full post with key sentences marked up
            # for analysis with simple string replacement...
            for summary_type in ['top_n_summary', 'mean_scored_summary']:
                post[summary_type +
                     '_marked_up'] = '<p>{0}</p>'.format(text)
                for s in post[summary_type]:

                    post[summary_type + '_marked_up'] = post[summary_type +
                                                             '_marked_up'].replace(s, '<strong>{0}</strong>'.format(s))

            return {
                'status': 201,
                'data': {
                    'summary': (' ').join(post[summary_type]),
                    'fullText': post[summary_type + '_marked_up']
                }}, 201

        return api.abort(404, "Expected text")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
