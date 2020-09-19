from django.shortcuts import render
# from website.camera import LiveWebCam
from django.http.response import StreamingHttpResponse,HttpResponse
from src1 import run_webcam
import cv2
# Create your views here.
# def gen(camera):
# 	while True:
# 		frame = camera.get_frame()
# 		yield (b'--frame\r\n'
# 				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def index(request):
    return render(request, 'index.html')

# def livecam_feed(request):
# 	return StreamingHttpResponse(gen(LiveWebCam()),
# 					content_type='multipart/x-mixed-replace; boundary=frame')

def input_video(request):
	a,b=run_webcam.main()
	return HttpResponse(cv2.imshow(a))						