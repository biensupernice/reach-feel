from flask import request
from flask_restful import Resource

from analyzer import analyze


class ResponseAnalysis(Resource):
    def post(self):
        actual_text = request.json['actual_text']
        optimal_text = request.json['optimal_text']
        sentiment_analysis, targetting_analysis = analyze(
            actual_text, optimal_text)
        return {
            'sentiment_analysis': sentiment_analysis,
            'targetting_analysis': targetting_analysis,
            'text': actual_text
        }
