import httpx
import os

import time

def test_predict():
    base_url = "http://localhost:8000"
    predict_url = f"{base_url}/predict"
    file_path = "data/teaser.jpg"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print(f"1. Submitting job to {predict_url}...")
    files = {'file': open(file_path, 'rb')}
    
    with httpx.Client(timeout=30) as client:
        try:
            # 1. Submit
            response = client.post(predict_url, files=files)
            if response.status_code != 200:
                print(f"Failed to submit job: {response.text}")
                return
            
            job_id = response.json()["job_id"]
            print(f"Job submitted. ID: {job_id}")
            
            # 2. Poll
            while True:
                status_url = f"{base_url}/jobs/{job_id}"
                status_res = client.get(status_url)
                job = status_res.json()
                
                print(f"Status: {job['status']} | Progress: {job['progress']}% | {job['message']}")
                
                if job['status'] == 'complete':
                    break
                elif job['status'] == 'failed':
                    print(f"Job failed: {job.get('error')}")
                    return
                
                time.sleep(1)
            
            # 3. Download
            print("Downloading result...")
            result_url = f"{base_url}/jobs/{job_id}/result"
            result_res = client.get(result_url)
            
            output_path = "backend/test_output.splat"
            with open(output_path, "wb") as f:
                f.write(result_res.content)
            
            print(f"Success! Saved output to {output_path}")
            print(f"File size: {os.path.getsize(output_path)} bytes")

        except Exception as e:
            print(f"Exception occurred: {e}")

def test_predict360():
    """Test the 360 panorama prediction endpoint."""
    base_url = "http://localhost:8000"
    predict_url = f"{base_url}/predict360"

    # Look for a 360 test image (equirectangular, ~2:1 aspect ratio)
    # Try common locations
    test_files = [
        "input/360_test.jpg",
        "input/panorama.jpg",
        "data/360_test.jpg",
    ]

    file_path = None
    for path in test_files:
        if os.path.exists(path):
            file_path = path
            break

    if not file_path:
        print("No 360 test image found. Tried:")
        for path in test_files:
            print(f"  - {path}")
        print("\nPlace a 360 equirectangular image (~2:1 aspect ratio) at one of these paths.")
        return

    print(f"1. Submitting 360 job to {predict_url}...")
    print(f"   Using file: {file_path}")
    files = {'file': open(file_path, 'rb')}

    with httpx.Client(timeout=300) as client:  # Longer timeout for 360 (6x inference)
        try:
            # 1. Submit
            response = client.post(predict_url, files=files)
            if response.status_code != 200:
                print(f"Failed to submit 360 job: {response.text}")
                return

            job_id = response.json()["job_id"]
            print(f"360 Job submitted. ID: {job_id}")

            # 2. Poll
            while True:
                status_url = f"{base_url}/jobs/{job_id}"
                status_res = client.get(status_url)
                job = status_res.json()

                print(f"Status: {job['status']} | Progress: {job['progress']}% | {job['message']}")

                if job['status'] == 'complete':
                    break
                elif job['status'] == 'failed':
                    print(f"360 Job failed: {job.get('error')}")
                    return

                time.sleep(2)  # Longer polling interval for 360

            # 3. Download
            print("Downloading 360 result...")
            result_url = f"{base_url}/jobs/{job_id}/result"
            result_res = client.get(result_url)

            output_path = "backend/test_output_360.splat"
            with open(output_path, "wb") as f:
                f.write(result_res.content)

            print(f"Success! Saved 360 output to {output_path}")
            print(f"File size: {os.path.getsize(output_path)} bytes")

        except Exception as e:
            print(f"Exception occurred: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "360":
        test_predict360()
    else:
        test_predict()
        print("\nTip: Run with '360' argument to test the 360 endpoint:")
        print("  python backend/test_api.py 360")
