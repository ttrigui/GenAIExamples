import time
from typing import Generator

import requests
from starlette.responses import StreamingResponse
from starlette.requests import Request
from ray import serve

serve.start(detached=True, 
            http_options={"host": "127.0.0.1", 
                          "port": 8888})
@serve.deployment
class StreamingResponder:
    def generate_numbers(self, max: int) -> Generator[str, None, None]:
        for i in range(max):
            yield str(i) + "\n"
            time.sleep(0.1)

    def __call__(self, request: Request) -> StreamingResponse:
        # max = request.query_params.get("max", "25")
        max = 25
        gen = self.generate_numbers(int(max))
        return StreamingResponse(gen, status_code=200, media_type="text/plain")


serve.run(StreamingResponder.bind())

while 1:
    time.sleep(100)
# r = requests.get("http://127.0.0.1:8888/", stream=True)
# start = time.time()
# r.raise_for_status()
# print("-"*40)
# print(type(r))
# for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
#     print(f"Got result {round(time.time()-start, 1)}s after start: '{chunk}'")