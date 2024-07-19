import pyttsx3
import speech_recognition as sr 
import datetime
import os 
import cv2
import random
import requests
import wikipedia
import pyjokes
import webbrowser
import pywhatkit as kit
import platform
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import socket 
import pyautogui
import instaloader
import sympy
"""import googletrans"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
import time
import imutils
import tensorflow as tf
from twilio.rest import Client
import numpy as np


engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 150)

# Text-to-speech part
def speak(audio, rate=160):
    engine.say(audio)
    print(audio)
    engine.runAndWait()





def send_sms(alert_message):
    account_sid = 'ACae17829af535161efc61fa13830da53a'  # Twilio Account SID
    auth_token = '4f20c831d6297f8336894dceb8f5ee64'    # Twilio Auth Token
    client = Client(account_sid, auth_token)
    alert_message = "Fire detected!"

    message = client.messages.create(
        body=alert_message,
        from_='+14156302581',
        to='+917903844254'
    )
    print(f"Message sent: {message.sid}")

def track_fire():
    # Fire color range in HSV
    Low = np.array([0,113,174])
    High = np.array([179,255,255])

    camera = cv2.VideoCapture(0)

    while True:
        (grabbed, frame) = camera.read()

        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, Low, High)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                alert_message = f"Fire detected at coordinates {center} with radius {radius}"
                print(alert_message)
                send_sms(alert_message)

                # Logic for identifying if the fire is significant enough to send a message
                if radius > 100:
                    print("Significant fire detected! Sending alert message.")
                    send_sms(alert_message)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    camera = cv2.VideoCapture(0)

    while True:
        _, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype('float32') / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)

            prediction = emotion_model.predict(roi_gray)
            max_index = int(np.argmax(prediction))
            emotion = emotion_labels[max_index]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()



def detect_objects():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), font, 1, color, 2)

        cv2.imshow("Object Detection", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

"""def get_system_info():
    system_info = {
        'System': platform.system(),
        'Node': platform.node(),
        'Release': platform.release(),
        'Version': platform.version(),
        'Machine': platform.machine(),
        'Processor': platform.processor(),
        'Architecture': platform.architecture(),
        'OS Name': os.name,
        'Current Directory': os.getcwd(),
        'User Home Directory': os.path.expanduser('~')
    }

    return system_info
"""

def scan_faces():
    algorithm = "haarcascade_frontalface_alt.xml" #accessing the pretrained model from the dir

    haar_cascade = cv2.CascadeClassifier(algorithm)

    cam = cv2.VideoCapture(0)
    while True:
        _, img = cam.read()
        text="No Face Detected"
        grayImage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converting the image to grayscale
        face = haar_cascade.detectMultiScale(grayImage,1.3,4)
        for (x,y,w,h) in face:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            text="Face Detected"
            
        print(text)
        cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("FACE DETECTION",img)
        key=cv2.waitKey(10)
        if key == 27:
            break
    cam.release()
    cv2.destoryAllWindows()

def obj_detection():
    cam = cv2.VideoCapture(0)
    time.sleep(1)
    firstframe = None
    area = 500

    while True:
        ret, img = cam.read()
        
        if not ret:
            break

        text = "Normal"
        img = imutils.resize(img, width=500)
        grayImg = cv2.cvtColor(img, cv2 .COLOR_BGR2GRAY)
        gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)

        if firstframe is None:
            firstframe = gaussianImg
            continue

        imgDiff = cv2.absdiff(firstframe, gaussianImg)
        threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]
        threshImg = cv2.dilate(threshImg, None, iterations=2)
        cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            if cv2.contourArea(c) < area:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0,255), 2)
            text = "Moving Object"

        print(text)
        cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Video Feed", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

def chat_with_bot():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print("Hey, I am Nebula. How can I help you?")
    speak("Hey, I am Nebula. How can I help you?")
    
    while True:
        speak_input = takecommand().lower()
        user_input = speak_input
        if not user_input:
            continue
        elif "bye" in user_input or "goodbye" in user_input:
            break
        
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        bot_response_ids = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
        bot_response = tokenizer.decode(bot_response_ids[0], skip_special_tokens=True)

        print(f"Bot: {bot_response}")
        speak(bot_response)

def take_screenshot():
    speak("Taking a screenshot")
    screenshot_path = "screenshot.png"
    pyautogui.screenshot(screenshot_path)
    speak("Screenshot taken. Here is the screenshot for you.")
    os.startfile(screenshot_path)

def search_on_maps():
    speak("Sure, please provide the location you want to search on Google Maps.")
    location_query = takecommand().strip()
    search_url = f"https://www.google.com/maps/search/{location_query}"
    webbrowser.open(search_url)
    speak(f"Searching Google Maps for {location_query}")

def read_news_headlines():
    speak("Fetching and reading the latest news headlines.")

    # Replace 'YOUR_API_KEY' with your actual News API key
    api_key = 'bc14d3f1253f4a3684f2fbc1929c4228'
    country = 'us'
    category = 'business'
    main_url = f'https://newsapi.org/v2/top-headlines?country={country}&category={category}&apiKey={api_key}'
    
    main_page = requests.get(main_url).json()
    
    if 'articles' in main_page:
        articles = main_page['articles']
        head = []
        day = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]

        for ar in articles:
            head.append(ar['title'])

        for i in range(len(day)):
            speak(f"Today's {day[i]} news is: {head[i]}")

def launch_website():
    speak("Sure, please provide the website URL.")
    website_url = takecommand().strip()
    webbrowser.open(website_url)
    speak(f"Launching the website {website_url}")

def translate_text():
    speak("Sure, please provide the text you want to translate.")
    text_to_translate = takecommand().strip()

    translator = googletrans.Translator()
    result = translator.translate(text_to_translate, dest='en')  # Translate to English

    speak(f"The translated text is: {result.text}")

def system_control():
    speak("Sure, what do you want to do with the system? You can say 'sleep', 'shut down', or 'restart'.")

    action = takecommand().lower()

    if "sleep" in action:
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")  # Puts the system to sleep
        speak("System is now in sleep mode.")
    elif "shut down" in action:
        os.system("shutdown /s /t 1")  # Shuts down the system after 1 second
        speak("Shutting down the system.")
    elif "restart" in action:
        os.system("shutdown /r /t 1")  # Restarts the system after 1 second
        speak("Restarting the system.")
    else:
        speak("I didn't understand the command. Please specify 'sleep', 'shut down', or 'restart'.")

def show_available_options():
    speak("I can assist you with the following tasks:")
    speak("1. Open Notepad")
    speak("2. Open Command Prompt")
    speak("3. Open Camera")
    speak("4. Send Email")
    speak("5. Play Music")
    speak("6. Get IP Address")
    speak("7. Search on Wikipedia")
    speak("8. Open YouTube")
    speak("9. Search on YouTube")
    speak("10. Open Google")
    speak("11. Send Message on WhatsApp")
    speak("12. Get Date and Time")
    speak("13. Check Weather")
    speak("14. Set Reminder")
    speak("15. Perform Calculations")
    speak("16. Switch Windows")
    speak("17. Get Location")
    speak("18. View Instagram Profile")
    speak("19. take a screenshot")
    speak("20. Search on Google Maps")
    speak("21. Read News Headlines")
    speak("22. Open Websites")
    speak("23. Translate Text")
    speak("24. Chat")
    speak("25. Show Available Options")
    speak("26. Exit")

def send_email():
    speak("Sure, let's compose an email.")
    
    speak("Please provide the sender's name.")
    sender_name = takecommand().strip()

    speak("Do you want to add any BCC? If yes, provide the email addresses separated by commas.")
    bcc_emails = takecommand().strip()

    speak("Do you want to add any CC? If yes, provide the email addresses separated by commas.")
    cc_emails = takecommand().strip()

    speak("What is the main mail about?")
    subject = takecommand().strip()

    speak("Please provide the message content.")
    message_content = takecommand().strip()

    try:
        # Set up the connection to the SMTP server
        smtp_server = "smtp.gmail.com"  # Update with your SMTP server details
        smtp_port = 587  # Update with your SMTP port
        smtp_username = "your_email@gmail.com"  # Update with your email address
        smtp_password = "your_email_password"  # Update with your email password

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)

        # Create the email message
        msg = MIMEMultipart()
        msg['From'] = sender_name
        msg['To'] = "recipient_email@example.com"  # Update with the recipient's email address
        msg['Bcc'] = bcc_emails
        msg['Cc'] = cc_emails
        msg['Subject'] = subject

        # Attach the message content
        msg.attach(MIMEText(message_content, 'plain'))

        # Send the email
        server.send_message(msg)

        speak("Email sent successfully.")
    except smtplib.SMTPException as e:
        speak(f"Error sending email: {str(e)}")
    except socket.gaierror as e:
        speak(f"Error resolving SMTP server address: {str(e)}")
    finally:
        try:
            # Close the connection to the SMTP server
            server.quit()
        except UnboundLocalError:
            pass  # Ignore if 'server' is not defined

def takecommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
        except sr.WaitTimeoutError:
            print("Timed out waiting for speech, please try again.")
            return "none"

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en_in')
        print(f"User said: {query}")
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand what you said.")
        return "none"
    except Exception as e:
        print(f"Error recognizing speech: {e}")
        speak("Say that again, please")
        return "none"

    return query

def wish():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour <= 12:
        speak("Good Morning")
    elif hour > 12 and hour < 16:
        speak("Good Afternoon")
    else:
        speak("Good Evening")

def calculate_expression(expression):
    try:
        result = sympy.sympify(expression)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def exit_command(query):
    goodbyes = ["bye", "take rest", "goodbye", "see you", "exit"]

    if any(phrase in query for phrase in goodbyes):
        speak("Goodbye! Take care.")
        exit()
        
def get_location():
    try:
        # Use the geocoder library to get location information based on IP address
        location = geocoder.ip('me')
        return location.latlng  # Returns latitude and longitude as a list
    except Exception as e:
        print(f"Error getting location: {e}")
        return None

def reply_to_greetings(query):
    greetings = ["hi", "hello", "who are you","what is you","what is your name"]
    if any(greet in query for greet in greetings):
        speak("Hello, I am Nebula.")
    elif "who are you" in query or "what is you" in query:
        speak("Hello. I am Nebula, your virtual assistant")



# Load the emotion detection model
"""emotion_model = load_model('emotion_model.hdf5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']"""

if __name__ == "__main__":
    wish()
    takecommand()
    while True:

        query = takecommand().lower()
        print(f"Recognized query: {query}")
        #logical building for tasks

        if "greet" in query or "hello" in query or "hi" in query:
            reply_to_greetings(query)

        elif "who are you" in query:
            speak("I am Nebula, your virtual assistant")
        
        elif "tell me a joke" in query or "share a joke" in query:
            joke = pyjokes.get_joke()
            speak(joke)

        elif "your favorite color" in query or "what color do you like" in query:
            speak("I don't have a favorite color, but I think ones and zeros look great together!")

        elif "do you dream" in query or "can you dream" in query:
            speak("No, I don't dream. But I'm always wide awake and ready to assist you!")

        elif "what is your hobby" in query or "do you have a hobby" in query:
            speak("My hobby is helping you and making your tasks easier. How can I assist you today?")

        elif "tell me something interesting" in query or "share an interesting fact" in query:
            speak("Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible!")

        elif "who made you" in query or "who is your owner" in query:
            speak("I was created by team Is Safe")
        
        elif "talk to me" in query or "let's talk" in query or "conversation" in query:
            chat_with_bot()

        elif "open notepad" in query:
            path = "C:\\Program Files\\WindowsApps\\Microsoft.WindowsNotepad_11.2311.35.0_x64__8wekyb3d8bbwe\\Notepad\\Notepad.exe"
            os.startfile(path)
      
        elif "open command prompt" in query:
            os.system("start")
        
        elif "open camera" in query:
            cap = cv2.VideoCapture(0)
            while True:
                ret, img = cap.read()
                cv2.imshow('Webcam', img)
                k = cv2.waitKey(50)
                if k == 27:
                    break
            cap.release()
            cv2.destroyAllWindows()
            
        elif "send email" in query:
            send_email()

        elif "play music" in query:
            music_dir = "C:\\Users\\satya\\Music"
            songs = os.listdir(music_dir)
            rd = random.choice(songs)
            os.startfile(os.path.join(music_dir, rd))
        
        elif "ip address" in query:
            ip = requests.get('https://api.ipify.org').text
            speak(f"Your IP address is {ip}")

        elif "wikipedia" in query:
            speak("Searching Wikipedia")
            query = query.replace("wikipedia", "")
            results = wikipedia.summary(query, sentences=2)
            speak("According to Wikipedia")
            speak(results, rate=120)
            print(results)

        elif "open youtube" in query:
            webbrowser.open("www.youtube.com")

        elif "search youtube" in query:
            query = query.replace("search youtube","")
            search_url = f"https://www.youtube.com/results?search_query={query}"
            webbrowser.open(search_url)
            speak(f"Searching Youtube for {query}")

        elif "open google" in query:
            speak("Sir, what should I search on Google?")
            search_query = takecommand().lower()

            if "nothing" in search_query or "just open" in search_query:
                webbrowser.open("https://www.google.com")
            else:
                webbrowser.open(f"https://www.google.com/search?q={search_query}")

        elif "send message" in query:
            speak("Sure, please provide the contact number.")
            contact_number = takecommand().strip()
            if not contact_number.startswith("+"):
                contact_number = "+91" + contact_number
            
            speak("What message would you like to send?")
            message = takecommand().strip()

            # Get the current time
            current_time = datetime.datetime.now()

            # Add 30 seconds to the current time
            send_time = current_time + datetime.timedelta(seconds=30)

            # Extract the hour and minute from the send_time
            time_hour = send_time.hour
            time_min = send_time.minute

            # Send the WhatsApp message
            kit.sendwhatmsg(contact_number, message, 2, 25)
            print(f"Sent message '{message}' to number {contact_number}")

        elif "date and time" in query:
            current_time = datetime.datetime.now()
            speak("Current time is")
            speak(current_time.strftime("%I:%M %p"))  # Format the time as HH:MM AM/PM

        elif "show available options" in query or "what can you do" in query:
            print("options is executing")
            """show_available_options()"""
                    
        elif "weather" in query:
            speak("Sure, please specify the city")
            city = takecommand().lower()
            # Use a weather API to get weather information for the specified city
            # Display or speak the weather details.
        
        elif "set reminder" in query:
            speak("Sure, what would you like to be reminded of?")
            reminder_text = takecommand().strip()
            # Use a scheduling mechanism to set a reminder and notify the user at the specified time.

        elif "calculate" in query:
            speak("Sure, please provide the calculation")
            calculation = takecommand().strip()
            result = calculate_expression(calculation)

            if isinstance(result, sympy.Basic):
                speak(f"The result of {calculation} is {result}")
            else:
                speak(result)

        elif "switch the windows" in query:
            pyautogui.keyDown("alt")
            pyautogui.press("tab")
            pyautogui.keyUp("alt")

        elif "get location" in query or "find my location" in query:
            location = get_location()
            if location:
                speak(f"Your current location is at latitude {location[0]} and longitude {location[1]}.")
            else:
                speak("Sorry, I couldn't determine your location.")
            
        elif "instagram profile" in query or "profile on instagram" in query:
            speak("Please enter the username")
            name = input("Enter username here:  ")
            webbrowser.open(f"www.instagram.com/{name}")
            speak(f"Here is the profile of the user {name}")

            speak("Would you like to download the profile picture of this account")
            condition = takecommand().lower()
            if "yes" in condition:
                mod = instaloader.Instaloader()
                mod.download_profile(name, profile_pic_only=True)
                speak("Download Complete")
            else:
                pass

        elif "take screenshot" in query:
            take_screenshot()
        elif "detect objects" in query:
            detect_objects()

        elif "search on maps" in query:
            search_on_maps()

        elif "fire" in query:
            track_fire()

        elif "read news headlines" in query:
            read_news_headlines()

        elif "launch website" in query:
            launch_website()

        elif "translate text" in query:
            translate_text()

        elif "take rest" in query or "goodbye" in query:
            exit_command()

        elif "Look for faces" in query or "scan faces" in query:
            scan_faces()

        elif "scan your surroundings" in query or "have a look" in query or "scan area" in query:
            obj_detection()
        
        elif "emotion detection" in query:
            emotion_detection()