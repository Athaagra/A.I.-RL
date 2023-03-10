#!/usr/bin/env python3

#
# Small server program for use with Cloud Computing 2023 Homework 1,
# LIACS, Leiden University.
#

import requests
import base64
import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

hash_url = "https://liacs.leidenuniv.nl/~rietveldkfd/tmp/cco2023hash"


class CCORequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/html")
            self.end_headers()
            self.wfile.write("\n".encode("ascii"))
            self.wfile.write("hello world.\n".encode("ascii"))
        elif self.path == "/ccomagic":
            # Try to retrieve hash
            try:
                r = requests.get(hash_url, allow_redirects=False)
                status = r.status_code
                if status != 200:
                    status = 500
            except requests.exceptions.RequestException:
                status = 500

            if status != 200:
                # bail out here
                self.send_error(501, "error computing hash")
                return

            # append timestamp
            tmp_hash = base64.b64decode(r.content.rstrip())
            tmp_hash += ":".encode('ascii')
            now = datetime.datetime.now().timestamp()
            tmp_hash += now.hex().encode('ascii')
            result = base64.b16encode(tmp_hash)

            self.send_response(200)
            self.send_header("Content-Type", "application/ascii")
            self.end_headers()
            self.wfile.write("\n".encode("ascii"))
            self.wfile.write(result)
        else:
            self.send_error(404, "we don't know about this!")



def run(server_class=HTTPServer, handler_class=CCORequestHandler):
    server_address = ('', 4004)
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()

def main():
    run()


if __name__ == '__main__':
    main()
