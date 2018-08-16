import json
import os
import urllib
import requests

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restful import reqparse, Api, Resource

from RedDetector import main

from pprint import pprint

app = Flask(__name__)
api = Api(app)
CORS(app)


parser = reqparse.RequestParser()
parser.add_argument('imageUrl')


class RednessDetection(Resource):
    def post(self):
        args = parser.parse_args()
        print(args)
        if args['imageUrl']:
         url = args['imageUrl']
        else:
          url = 'https://cdn2.stylecraze.com/wp-content/uploads/2013/03/Cystic-Acne-%E2%80%93-What-Is-It-And-How-To-Cure-It-2.jpg'
        r = requests.get(url, allow_redirects=True)
        print(r)
        open('99.jpg', 'wb').write(r.content)
        return (main('99.jpg'))


api.add_resource(RednessDetection, '/redness')

if __name__ == '__main__':
    app.run(debug=True,  host = os.getenv("IP","0.0.0.0"),port = int (os.getenv('PORT', 33507)))