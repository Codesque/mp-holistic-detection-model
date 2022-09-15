
import mediapipe as mp 
import cv2 
import time 

class Holistic_Face_Detection: 

    def __init__(self) -> None:
        self.initialise_model() 
        self.image_manuplation = None  # Takes frame , width ,height as an argument 
        self.landmarks = None  
        self.displayFPS = True   
        self.drawFace = True 
        self.drawHands = False 
        self.FPSColor = (30,204,102) 
        self.caption = "Holistic Face Detection" 

    


    def initialise_model(self ,min_detect_conf = 0.5 , min_track_conf = 0.5 , complexity = 1 , image_mode = False , smooth_landmark = True  ): 
        self.holistic_solution = mp.solutions.holistic 
        self.holistic_model = self.holistic_solution.Holistic(
            static_image_mode = image_mode , # Are you going to detect faces in  images or video streams? 
            model_complexity = complexity , #the complexity of the pose landmark model: 0, 1, or 2.model complexity increases landmark accuracy and latency
            smooth_landmarks = smooth_landmark,# reduce the jitter in the prediction by filtering pose landmarks across different input images
            min_detection_confidence = min_detect_conf , #minimum confidence value with which the detection from the person-detection model 
            min_tracking_confidence = min_track_conf #minimum confidence value with which the detection from the landmark-tracking model 
        )    

        self.brush = mp.solutions.drawing_utils   


    def detectLandmarks(self , camera : cv2.VideoCapture , frame_width = 800 , frame_height = 600 ): 
        success , frame = camera.read()   
        frame = cv2.resize(frame ,(frame_width , frame_height)) 
        rgb_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)  

        rgb_frame.flags.writeable = False  
        landmarks = self.holistic_model.process(rgb_frame)  
        rgb_frame.flags.writeable = True   

        bgr_frame = cv2.cvtColor(rgb_frame , cv2.COLOR_RGB2BGR) 

        return landmarks , bgr_frame 



        
    
    def drawLandmarks(self ,bgr_frame , drawFace = True , drawHands = False,color1 = (30,102,204) , color2 = (204,102,30)): 

        #print(dir(self.holistic_solution))
        if drawFace :  
            self.brush.draw_landmarks( 

                bgr_frame , 
                self.landmarks.face_landmarks, 
                self.holistic_solution.FACEMESH_TESSELATION , 
                self.brush.DrawingSpec(
                    color = color1, 
                    thickness = 1 , 
                    circle_radius = 1
                ) , 
                self.brush.DrawingSpec(
                    color = color2, 
                    thickness = 1 , 
                    circle_radius = 1

                )
            ) 

        if drawHands : 
            # Right Hand : 
            self.brush.draw_landmarks(
                bgr_frame,
                self.landmarks.right_hand_landmarks,
                self.holistic_solution.HAND_CONNECTIONS 
            ) 

            # Left Hand : 
            self.brush.draw_landmarks(
                bgr_frame,
                self.landmarks.left_hand_landmarks,
                self.holistic_solution.HAND_CONNECTIONS
            )

        
        
    def calculateFPS(self , previous_timer :int , fps = 0 ):  
         
        fps =  (time.time() - previous_timer)**-1  
        return fps 



    def activate(self):  
        """
        Algorithm of Detecting Face and Hand Landmarks : 
        1. Capture frames with OpenCV 
        2. CV2 outputs frames as BGR . resize them and turn them to RGB  
        3. Predictions that made by the model are saved as the attributes of the upcoming output. Use them to access landmarks 
        4. Draw the landmarks 
        5. Display the image 
        """

        camera = cv2.VideoCapture(0) 
        previous_time = 0 
        running = True 
        while running and camera.isOpened() :  
            
            self.landmarks , bgr_frame = self.detectLandmarks(camera) 
            self.drawLandmarks(bgr_frame , self.drawFace , self.drawHands)  
          
            #if self.displayFPS: 
            fps = self.calculateFPS(previous_time)    
            cv2.putText(bgr_frame, str(int(fps))+" FPS", (10, 70),cv2.FONT_HERSHEY_COMPLEX, 1, self.FPSColor, 2)   

            cv2.imshow(self.caption ,bgr_frame) 

            if cv2.waitKey(5) > 0 : 
                running = False  
            previous_time = time.time()

        camera.release()
        cv2.destroyAllWindows() 
        


if __name__ == "__main__": 
    program = Holistic_Face_Detection()  
    program.drawHands = True 
    program.activate()