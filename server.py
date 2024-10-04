import http.server
import socketserver
import matlab.engine
import json
import numpy as np
import matlab

PORT = 8000

# Start the MATLAB engine once, globally
data = None  # Global variable to store POST data
eng = matlab.engine.start_matlab()

class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        global data

        try:
            # Get the content length and read the data from the request
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(post_data)

            # Extract data (if any, for prediction job these might be missing)
            X_train = np.array(data.get('X_train', []))
            X_pred = np.array(data.get('X_pred', []))
            y_train = np.array(data.get('y_train', []))

            # Print the received data shapes
            print("X_train shape:", X_train.shape)
            print("y_train shape:", y_train.shape)
            print("X_pred shape:", X_pred.shape)

            # Extract parameters
            pop_size = data.get('pop_size', 100)
            num_gen = data.get('num_gen', 50)
            tournament_size = data.get('tournament_size', 5)
            timeout = data.get('timeout', 60)

            # Print other parameters
            print("pop_size:", pop_size)
            print("num_gen:", num_gen)
            print("tournament_size:", tournament_size)
            print("timeout:", timeout)

            # Run MATLAB script based on the job type
            result = run_matlab_script(X_train, y_train, X_pred, pop_size, num_gen, tournament_size, timeout)

            response = {
                'y_pred_train': result.get('y_pred_train', None),
                'y_pred_pred': result.get('y_pred_pred', None)
            }
            
            # Send a response back to the client
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))

        except Exception as e:
            print(f"Error: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))

def run_matlab_script(X_train, y_train, X_pred, pop_size, num_gen, tournament_size, timeout):
    # Convert arrays to MATLAB format if provided
    X_train_matlab = matlab.double(X_train.tolist()) if X_train.size > 0 else matlab.double([])
    y_train_matlab = matlab.double(y_train.tolist()) if y_train.size > 0 else matlab.double([])
    X_pred_matlab = matlab.double(X_pred.tolist()) if X_pred.size > 0 else matlab.double([])

    try:
        # Call the MATLAB function with the correct number of arguments
        result = eng.gptips2(X_train_matlab, y_train_matlab, X_pred_matlab, 
                             pop_size, num_gen, tournament_size, timeout, nargout=1)

        print("Model fitted successfully")

        # Assuming 'result' is a structure, extract the predicted fields
        y_pred_train = np.array(result['y_pred_train']).tolist() if 'y_pred_train' in result else None
        y_pred_pred = np.array(result['y_pred_pred']).tolist() if 'y_pred_pred' in result else None

        return {
            'y_pred_train': y_pred_train,
            'y_pred_pred': y_pred_pred
        }

    except Exception as e:
        print(f"Error during MATLAB execution: {e}")
        raise


if __name__ == "__main__":
    # Allow the reuse of address if the server is restarted quickly
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), RequestHandler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()
