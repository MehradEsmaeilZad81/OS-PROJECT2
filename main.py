import ast
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List, TupleF
import numpy as np


class Worker:
    def __init__(self):
        self.is_busy = False

    # Method to calculate the dot product of a row and a column. Return zero if the worker is busy.
    def process(self, x: List[float], y: List[float]) -> Tuple[float, object]:
        if self.is_busy:
            return 0, "This worker is currently busy."
        self.is_busy = True
        val = np.dot(x, y)
        self.is_busy = False
        return val, None


class Server(BaseHTTPRequestHandler):
    def __init__(self, request, client_addr, server):
        # Set the default number of workers
        num_workers = 5
        self.num_workers = num_workers

        # Create worker instances based on the number of workers
        self.workers = [Worker() for _ in range(num_workers)]

        # Maintain a set of free workers (acting as a semaphore)
        self.free_workers = set(self.workers)

        # RLock used as a mutex lock
        self.lock = threading.RLock()

        # Store the responses in a dictionary
        self.responses = dict()

        # Call the parent class constructor
        super().__init__(request, client_addr, server)

    def do_POST(self):
        if self.path == "/setNumberOfWorkers":
            self.set_num_workers()
        elif self.path == "/multiply":
            self.process()
        else:
            self.send_error(404)

    def set_num_workers(self):
        # Get the data from the request
        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)
        dict_str = body.decode("UTF-8")
        data = ast.literal_eval(dict_str[1:-1])

        if "num_workers" not in data:
            # If the number of workers is not provided in the request, send an error response
            self.send_error(
                400, "Please provide the number of workers in your request.")
            return
        num_workers = int(data["num_workers"])

        # Lock the section to ensure thread safety when changing the number of workers
        with self.lock:
            print(
                f"Number of workers updated: {self.num_workers} -> {num_workers}")
            self.num_workers = num_workers

        # Store the new number of workers in the responses dictionary
        self.responses[self.path] = num_workers

        # Send a success response with the new number of workers to the client
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response_data = json.dumps(
            {'new number of workers': self.responses[self.path]})
        self.wfile.write(response_data.encode())

    def process(self):
        # Get the data from the request
        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)
        dict_str = body.decode("UTF-8")
        data = ast.literal_eval(dict_str[1:-1])

        if 'A' not in data or 'B' not in data:
            # If matrices A and B are not provided in the request, send an error response
            self.send_error(
                400, "Please provide A and B matrices in your request.")
            return
        # Initialize the input matrices. Calculate the transpose of B for easier further calculations.
        A = np.array(data["A"])
        B_transpose = np.array(data["B"]).T

        if A.shape[1] != B_transpose.shape[1]:
            # If the dimensions of the matrices are not compatible for multiplication, send an error response
            self.send_error(
                400, "You can't multiply two matrices with these dimensions. Consider changing them.")
            return
        n, m = A.shape[0], B_transpose.shape[0]
        resp = np.zeros((n, m))
        # Create a lock for when a worker wants to modify a cell in the final matrix
        resp_mutex = threading.RLock()
        # Number of cells calculated in the final matrix
        resp_count = 0
        # Total number of cells in the final matrix
        resp_total = n * m
        # An event to let the program know when the calculations are over
        resp_event = threading.Event()

        # Calculate the cell in the i-th row and j-th column of the final matrix using the worker w
        def process_task(w: Worker, i: int, j: int):
            # Multiply the i-th row of A and j-th column of B
            val, err = w.process(A[i], B_transpose[j])
            if err is not None:
                # Handle the error if encountered during the calculation
                print(err)
            else:
                # Acquire the lock. Only one worker can access and modify this cell.
                with resp_mutex:
                    resp[i][j] = val
                    nonlocal resp_count
                    resp_count += 1
                    # Check if we have calculated all cells in the final matrix.
                    # If yes, then signal the program to create the response.
                    if resp_count == resp_total:
                        resp_event.set()
                self.free_workers.add(w)

        for i in range(n):
            for j in range(m):
                # Boolean flag to check if any worker is available to perform the calculation.
                done = False
                # The loop will continue until a free worker is available to assign a calculation.
                while not done:
                    # Acquire the lock when choosing a new worker for the calculation.
                    with self.lock:
                        if self.free_workers:
                            w = self.free_workers.pop()
                            done = True
                        else:
                            w = None
                    if done:
                        # Assign the worker and let it start its job in a separate thread
                        thread = threading.Thread(
                            target=process_task, args=(w, i, j))
                        thread.start()
                    else:
                        # No free worker found, so sleep a little before checking again for available workers.
                        time.sleep(0.001)

        # Wait for the signal of completion of the calculations to create a response
        resp_event.wait()

        # Store the final answer in the responses dictionary
        self.responses[self.path] = resp.tolist()

        # Send a success response with the final result to the client
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response_data = json.dumps({'result': self.responses[self.path]})
        self.wfile.write(response_data.encode())


# Run the server
def run(server_class=HTTPServer, handler_class=Server, port=5000):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting the server on port {port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
