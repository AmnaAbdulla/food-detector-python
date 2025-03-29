from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests

app = Flask(__name__)

def url_to_image(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

@app.route("/classify", methods=["POST"])
def classify_food():
    data = request.json
    image_url1 = data.get("image_url1")
    image_url2 = data.get("image_url2")
    
    result1 = classify_food_image(image_url1)
    result2 = classify_food_image(image_url2, is_second_image=True)

    final_result = "True" if result1.startswith("True") and result2.startswith("True") else "False"

    return jsonify({
        "final_result": final_result
    })

def classify_food_image(image_url, is_second_image=False):
    image = url_to_image(image_url)
    if image is None:
        return "Invalid: Image not found!"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=30, minRadius=50, maxRadius=350)
    
    if circles is None:
        return "False (No Plate Detected)"
    
    circles = np.uint16(np.around(circles))
    (x, y, r) = circles[0][0]
    
    plate_mask = np.zeros_like(gray)
    cv2.circle(plate_mask, (x, y), r, 255, -1)
    plate_area = cv2.bitwise_and(image, image, mask=plate_mask)
    
    hsv = cv2.cvtColor(plate_area, cv2.COLOR_BGR2HSV)
    
    lower_food = np.array([5, 50, 50])
    upper_food = np.array([35, 255, 255])
    food_mask = cv2.inRange(hsv, lower_food, upper_food)
    
    plate_pixel_count = np.sum(plate_mask > 0)
    food_pixel_count = np.sum(food_mask > 0)
    food_coverage = food_pixel_count / plate_pixel_count
    
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_coverage = np.sum(yellow_mask > 0) / plate_pixel_count
    
    if is_second_image:
        if food_coverage < 0.2:
            return True
        else:
            return False
    else:
        if yellow_coverage > 0.5:
            return False
        if food_coverage > 0.2:
            return True
        else:
            return False

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
