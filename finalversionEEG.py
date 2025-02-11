#eeg experiment with 4 tasks (shape, line, letter, color) with 2 feedback conditions(comparitive and difficulty)
#look at influence of feedback on confidence in 4 tasks in EEG
#eeg markers 
pilot = 0 #put to 0 for the real experiment!

#import modules
from psychopy import visual as vis
from psychopy import event, core, logging, gui, data
import os, pandas,random, csv, itertools
from os      import listdir
from os.path import isfile, join
import numpy as np
from time import sleep
from statistics import mean
import random
import serial

  # EEG trigger on = 1, off = 0 -> turn off while not connected to EEG or ERROR
trigger = 0
my_directory  = os.getcwd()

##Create a Data folder if it doesn't exist yet
if os.path.isdir('eeg_test') == False:
    os.mkdir('eeg_test')

#GUI
if pilot:
    sub = 0;age = 30;gender = 'M';handedness = 'R'
else:
    info           = {"sub": 0,"gender":['V','M','X'],"age": 0,"handedness":['R','L']}
    myDlg = gui.DlgFromDict(dictionary = info, title = "eeg_test",show=True)
    sub=info['sub'];age=info['age'];gender=info['gender'];handedness=info['handedness'];
    file_name = "eeg_test/eeg_test_sub%d" %(sub) + ".csv"
    if os.path.isfile(file_name):
        print('This subject number already exists!')
        core.quit()

# design
if pilot:
    nb_training_blocks  = 5
    nb_main_blocks      = 1
    nb_total_blocks     = nb_training_blocks + nb_main_blocks
    nb_training_trials  = 12
    nb_trials           = 12
    time_fb             = 0.5   

else:
    nb_training_blocks  = 5
    nb_main_blocks      = 3
    nb_total_blocks     = nb_training_blocks + nb_main_blocks
    nb_training_trials  = 18
    nb_trials           = 60 #needs to be a multitude of 12 and 6 (3 difficulty conditions x 4 or 2 trial options)
    time_fb             = 0.5   

## EEG Triggers
# the following code depends on the EEG device in the lab and should be changed accordingly!
# Check what COM via "start" -> "device manager"

if trigger:
    serialport = serial.Serial("COM3", baudrate=115200) #matches with the baudrate at CogTex EEG's  
    # Example list of triggers
    Trig_Begin_Experiment = b'A'  # 65 --> label in ActiView
    Trig_Begin_Target = b'G'  # 71
    Trig_Begin_Feedback = b'H'  # 72
    Trig_left_ERP = b'J'  # 74
    Trig_right_ERP = b'N'  # 78

    Trig_cj1 = b'M'  # 77
    Trig_cj2 = b'C'  # 67
    Trig_cj3= b'E'  # 69
    Trig_cj4  = b'I'  # 73
    Trig_cj5 = b'L'  # 76
    Trig_cj6 = b'K'  # 75
    
    # Define function to send trigger
    # We will pass the trigger type (e.g. Trig_Begin_Trial) to this
    # function whenever we want to send a trigger

    def sendTrigger(Trigger):
        serialport.write(Trigger)
        serialport.flush()

##clock
clock         = core.Clock()
##timing
time_cross = .75   ##time fixation cross
time_stim = .2 #time stimulus on screen
##visual features
#win   = vis.Window(size=[1900,1060],color='black',allowGUI = False , units ='norm',fullscr=False)
win = vis.Window(color='black', units = 'norm', fullscr = True)
win.mouseVisible = False 
scr_width = win.size[0]
scr_height = win.size[1]
#fix   = vis.Rect(win,width=25/scr_width, height=14.07/scr_height,fillColor=[-1,1,-1],lineColor=[-1,1,-1])
fix = vis.TextStim(win, text = '+', color = [-1,1,-1], height = 0.08) 
## Dots and rectangles
rect = vis.Rect(win,size=(0.55,0.95),pos=(0,0),lineColor="white", fillColor = 'black') #Centered rectangle - Pierre
## Response labels
remheight = 0.06
lettersheight = 0.05
wrapwidth = 1.2 
instrheight = 1.5
instrwidth = 1.6



# Confidence labels, counterbalance the order between participants
#cj_text= vis.TextStim(win,text='Hoe zeker ben je dat je de juiste keuze hebt gemaakt?', pos=(0,0.5), height=.08)

cj_labels = vis.TextStim(win,text='Zeker fout      Vermoed fout      Gok fout      Gok juist      Vermoed juist      Zeker juist', pos=(0,0.8),height=.04, color = 'gray')

# feedback
good  = vis.TextStim(win, text = "Juist!", color = 'green')
bad   = vis.TextStim(win, text = "Fout!", color = 'red')



## FEEDBACK: vertical rectangle 
bigrect       = vis.Rect(win, fillColor= None, lineWidth = 5, lineColor='white', size = (0.2,1.5))
midline       = vis.Rect(win, fillColor='white', lineColor='white', size=(0.2,0.01), pos = (0,0))

average_label = vis.TextStim(win, text = "Gemiddelde", color = 'white', pos = (0.2,0), height = 0.04)
best_label    = vis.TextStim(win, text = "Beste prestatie", color = 'white', pos = (0.2,0.73),height = 0.04)
worst_label   = vis.TextStim(win, text = "Slechtste prestatie", color = 'white', pos = (0.22,-0.73),height = 0.04)

fb_nexttext  =  vis.TextStim(win, text = "(Druk C of N om verder te gaan)", color = 'white', pos = (0,-0.9), height = 0.03)


def givefeedback(fb_cond,block):
    if fb_cond == 'negativefb':
        # one time medium feedback to conceal the manipulation, but still <50% 
        if block == 1:
            score = np.random.choice(range(32,37),1)[0]
            yourscorecolour = (.9,.9,.9)
            verbalscore = 'Middelmatig: {0}%'.format(score)
        else: 
            score = np.random.choice(range(5,31),1)[0]
            colour_incr = (score-5)*0.04
            yourscorecolour = (1,-1+colour_incr,-1+colour_incr) # from red to white
            verbalscore = 'Slechtste {0}%'.format(score)       
    elif fb_cond == 'positivefb':
        # one time medium feedback to conceal the manipulation, but still >50% 
        if block == 2:
            score = np.random.choice(range(63,68),1)[0]
            yourscorecolour = (.9,.9,.9)
            verbalscore = 'Middelmatig: {0}%'.format(score)
        else:
            score = np.random.choice(range(70,96),1)[0]
            colour_incr = (95-score)*0.04
            yourscorecolour = (-1+colour_incr, 1, -1+colour_incr)  # from green to white
            verbalscore = 'Beste {0}%'.format(score)
     
    scoretext = vis.TextStim(win, text = verbalscore, 
                                     color = yourscorecolour, 
                                     height = 0.1, 
                                     pos = (0.1,0.9))
    smallrect = vis.Rect(win, fillColor = yourscorecolour, 
                                 lineColor=None,
                                 size=(0.2, score*0.015), 
                                 pos = (0,-0.75+(score*0.015)/2))
    smallrect.draw()
    bigrect.draw()
    average_label.draw()
    best_label.draw()
    worst_label.draw()
    midline.draw()
    scoretext.draw()
    fb_nexttext.draw()
    win.flip()
    core.wait(1)
    event.waitKeys(keyList = ['c', 'n'])
    return score;

# tasks
A = 'color'
B = 'shape'
C = 'letter'
D = 'orientation'
# feedback conditions
N = 'negativefb'
P = 'positivefb'
#training difficulty
E = 'easy'
H = 'hard'



## Counterbalance taks, feedback cond 
if sub%8 == 0:
    fb_cond = [P,N,E,H]
if sub%8 == 1:
    fb_cond = [P,N,H,E]
if sub%8 == 2:
    fb_cond = [N,P,H,E]
if sub%8 == 3:
    fb_cond = [N,P,E,H]
if sub%8 == 4:
    fb_cond = [E,H,P,N]
if sub%8 == 5:
    fb_cond = [E,H,N,P]
if sub%8 == 6:
    fb_cond = [H,E,N,P]
if sub%8 == 7:
    fb_cond = [H,E,P,N]
        
if 0 < sub <= 8:
    task_order = [A,B,C,D]
if 8 < sub <= 16:
    task_order = [D,A,B,C]
if 16 < sub <= 24:
    task_order = [C,D,A,B]
if 24 < sub <= 32:
    task_order = [B,C,D,A]


    
##define keys
choice_keys = ['c','n','escape'] #left, right, escape
cj_keys = ['1','2','3','8','9','0']

##define coordinates for shape and letter tasks
xcoord  = np.around(np.linspace(-0.24*scr_width/2, 0.24/2*scr_width, num=10),2)
xcoords = np.repeat(xcoord, 15)
ycoord  = np.around(np.linspace(-0.44*scr_height/2, 0.44*scr_height/2, num=15),2)
ycoords = np.tile(ycoord, 10)
coordinates_letter = list(zip(xcoords, ycoords))

xcoord  = np.around(np.linspace(-0.24*scr_width/2, 0.24/2*scr_width, num=12),2)
xcoords = np.repeat(xcoord, 20)
ycoord  = np.around(np.linspace(-0.44*scr_height/2, 0.44*scr_height/2, num=20),2)
ycoords = np.tile(ycoord, 12)
coordinates_shape = list(zip(xcoords, ycoords))

#coordinates 
xcoord  = np.around(np.linspace(-0.24*scr_width/2, 0.24/2*scr_width, num=10),2)
xcoords = np.repeat(xcoord, 15)
ycoord  = np.around(np.linspace(-0.44*scr_height/2, 0.44*scr_height/2, num=15),2)
ycoords = np.tile(ycoord, 10)
coordinates_orientation = list(zip(xcoords, ycoords))
coordinates = {'shape':coordinates_shape,'letter':coordinates_letter, 'orientation': coordinates_orientation}

##Task instructions
colour_instrfb = vis.ImageStim(win,image=my_directory+'\\1B_Colour_instr.JPG',pos=(0,0), interpolate=True)
shape_instrfb = vis.ImageStim(win,image=my_directory+'\\1B_Shape_instr.JPG',pos=(0,0), interpolate=True)
letter_instrfb = vis.ImageStim(win,image=my_directory+'\\1B_Letter_instr.JPG',pos=(0,0), interpolate=True)
orientation_instrfb = vis.ImageStim(win,image=my_directory+'\\1B_orientation_instr.JPG',pos=(0,0), interpolate=True)
task_instr_fb = {'color':colour_instrfb,'shape':shape_instrfb,'letter':letter_instrfb, 'orientation': orientation_instrfb}

colour_instrdf = vis.ImageStim(win,image=my_directory+'\\Colour_instr.JPG',pos=(0,0),interpolate=True)
shape_instrdf = vis.ImageStim(win,image=my_directory+'\\Shape_instr.JPG',pos=(0,0),interpolate=True)
letter_instrdf = vis.ImageStim(win,image=my_directory+'\\Letter_instr.JPG',pos=(0,0),interpolate=True)
orientation_instrdf = vis.ImageStim(win,image=my_directory+'\\orientation_instr.JPG',pos=(0,0),interpolate=True)
task_instr_diff = {'color':colour_instrdf,'shape':shape_instrdf,'letter':letter_instrdf, 'orientation': orientation_instrdf}

#response labels 
red_lab  = vis.TextStim(win, text = "R", color = 'red', pos = (-0.7,0), height = remheight)
blue_lab = vis.TextStim(win, text = "B", color = 'blue', pos = (0.7,0), height = remheight)

#Letter
x_lab  = vis.TextStim(win, text = "A", color = 'white', pos = (-0.7,0), height = remheight)
o_lab = vis.TextStim(win, text = "B", color = 'white', pos = (0.7,0), height = remheight)

#Shape
square_lab  = vis.Rect(win, units = "pix", lineColor = 'white', fillColor = 'black', pos = (-0.7*scr_width/2,0), size = remheight*scr_width/2)
circle_lab = vis.Circle(win,units = "pix", lineColor = 'white', fillColor = 'black',pos = (0.7*scr_width/2,0), size = remheight*scr_width/2)

#orientation
horizontal_lab = vis.TextStim(win, text = '—', color = 'white', pos = (-0.70,0), height = remheight)
vertical_lab = vis.TextStim(win, text = '|', color = 'white', pos = (0.70,0), height = remheight )

resp_labels_left = {"color":red_lab,"shape":square_lab,"letter":x_lab, 'orientation':horizontal_lab}
resp_labels_right = {"color":blue_lab,"shape":circle_lab,"letter":o_lab, 'orientation':vertical_lab}
## Make and show Instructions
slide1 = vis.ImageStim(win, image = my_directory+'/instructions1.jpeg',pos=(0,0),interpolate=True)
slide2 = vis.ImageStim(win, image = my_directory+'/instructions_start.JPG',pos= (0,0))
#slide3 = vis.ImageStim(win, image = my_directory+'/instructions3.jpeg',pos = (0,0))

if trigger:
   sendTrigger(Trig_Begin_Experiment)
instruction = [slide1,slide2]
for instr in range(len(instruction)):
    #if trigger:
      # sendTrigger(Trig_show_instruction) 
    runinstr = instruction[instr]
    runinstr.draw()
    win.flip()
    sleep(1)
    k = event.waitKeys(keyList = ['c', 'n'])

###MAKE A DATA FILE
#TrialHandler: make a data file
file_name = "eeg_test/eeg_test_sub%d" %(sub)
info           = {"sub": sub,"age": age, "gender": gender, "handedness": handedness}
thisExp = data.ExperimentHandler(dataFileName = file_name,extraInfo=info)

###EXPERIMENT START 
for task in range(len(task_order)):
        task_name = task_order[task]
        fb_name = fb_cond[task]
        #Show task intruction
        if fb_name == 'negativefb' or fb_name == 'positivefb':
            #different instructions depending on which feedback type is first! 
            if task == 2:
                slidemid = vis.ImageStim(win, image = my_directory+'/instructiemidden.jpg',pos= (0,0),interpolate=True)
                slide3 = vis.ImageStim(win, image = my_directory+'/instructions2_changecom.jpg',pos= (0,0),interpolate=True)
                slide4 = vis.ImageStim(win, image = my_directory+'/instructions3.jpeg',pos = (0,0),interpolate=True)
                slide5 = task_instr_fb[task_name]
                instructions = [slidemid, slide3,slide4,slide5]
                for instr in range(len(instructions)):
                    runinstr = instructions[instr]
                    runinstr.draw()
                    win.flip()
                    #if trigger:
                       #sendTrigger(Trig_show_instruction) 
                    sleep(1)
                    k = event.waitKeys(keyList = ['c', 'n'])        
            else: 
                slide3 = vis.ImageStim(win, image = my_directory+'/instructions2_com.jpg',pos= (0,0),interpolate=True)
                slide4 = vis.ImageStim(win, image = my_directory+'/instructions3.jpeg',pos = (0,0),interpolate=True)
                slide5 = task_instr_fb[task_name]
                instructions = [slide3,slide4,slide5]
                for instr in range(len(instructions)):
                    runinstr = instructions[instr]
                    runinstr.draw()
                    win.flip()
                    #if trigger:
                       #sendTrigger(Trig_show_instruction) 
                    sleep(1)
                    k = event.waitKeys(keyList = ['c', 'n'])     
        elif fb_name == 'easy' or fb_name == 'hard':
            if task == 2: 
                slidemid = vis.ImageStim(win, image = my_directory+'/instructiemidden.jpg',pos= (0,0),interpolate=True)
                slide3 = vis.ImageStim(win, image = my_directory+'/instructions2_changediff.jpg',pos= (0,0),interpolate=True)
                slide4 = vis.ImageStim(win, image = my_directory+'/instructions3.jpeg',pos = (0,0),interpolate=True)
                slide5 = task_instr_diff[task_name]
                instructions = [slidemid, slide3,slide4,slide5]
                for instr in range(len(instructions)):
                    runinstr = instructions[instr]
                    runinstr.draw()
                    win.flip()
                  #  if trigger:
                       #sendTrigger(Trig_show_instruction) 
                    sleep(1)
                    k = event.waitKeys(keyList = ['c', 'n'])
            else: 
                slide3 = vis.ImageStim(win, image = my_directory+'/instructions2_diff.jpg',pos= (0,0),interpolate=True)
                slide4 = vis.ImageStim(win, image = my_directory+'/instructions3.jpeg',pos = (0,0),interpolate=True)
                slide5 = task_instr_diff[task_name]
                instructions = [slide3,slide4,slide5]
                for instr in range(len(instructions)):
                    runinstr = instructions[instr]
                    runinstr.draw()
                    win.flip()
                  #  if trigger:
                       #sendTrigger(Trig_show_instruction) 
                    sleep(1)
                    k = event.waitKeys(keyList = ['c', 'n'])
        for block in range(nb_total_blocks):            
            if block < nb_training_blocks: #5
                running = "trainingphase"
                trials = nb_training_trials  #18
               #if difficulty condition, make sure difficulty doens't change within trai
                if fb_name == 'easy':
                   condition = np.repeat((0,3),trials/2) #this function repeats the number 0 and 3 for the amount of trials we need, which we then later use to define difficulty and the correct answer 
                   random.shuffle(condition)
                   accuracy = np.zeros(shape=(trials,1))
                elif fb_name == 'hard':
                    condition = np.repeat((2,5),trials/2)
                    random.shuffle(condition)
                    accuracy = np.zeros(shape=(trials,1))
                elif fb_name == 'negativefb':
                    #for other condition it can change within trail 
                    condition = np.repeat(range(0,6),trials/6)
                    random.shuffle(condition)
                    accuracy = np.zeros(shape=(trials,1))
                elif fb_name == 'positivefb':
                    condition = np.repeat(range(0,6),trials/6)
                    random.shuffle(condition)
                    accuracy = np.zeros(shape=(trials,1))
            else:
                #show confidence instructions again
                if block == 5:
                    slide6 = vis.ImageStim(win, image = my_directory+'/instructions4.jpeg',pos = (0,0),interpolate=True)
                    slide6.draw()
                    win.flip()
                    sleep(1)
                    k = event.waitKeys(keyList = ['c', 'n'])
                else:
                    slide7 = vis.ImageStim(win, image = my_directory+'/instructions5.jpeg', pos = (0,0), interpolate = True)
                    slide7.draw()
                    win.flip()
                    sleep(1)
                    k = event.waitKeys(keyList = ['c','n'])
                 


                
                #actual task 
                
                trials = nb_trials
                running = "main"           
                condition = np.repeat(range(0,6),trials/6)  
                random.shuffle(condition)
                accuracy = np.zeros(shape=(trials,1))
            for trial in range(trials): 
                if condition[trial] == 0 or condition[trial] == 3:
                   if task_name == 'color':
                       difficulty = np.random.choice(range(52,54),1)[0]
                       difflevel = 'easy'
                   elif task_name == 'letter':
                       difficulty = np.random.choice(range(63,65),1)[0]
                       difflevel = 'easy'
                   elif task_name == 'shape':
                       difficulty = np.random.choice(range(61,63),1)[0]
                       difflevel = 'easy'
                   elif  task_name == 'orientation':   
                       difficulty = np.random.choice(range(57,59),1)[0]
                       difflevel = 'easy'
                elif condition[trial] == 1 or condition[trial] == 4 :
                    if task_name == 'color':
                        difficulty = np.random.choice(range(46,48),1)[0]
                        difflevel = 'medium'
                    elif task_name == 'shape':
                        difficulty = np.random.choice(range(56,58),1)[0]
                        difflevel = 'medium'
                    elif task_name == 'letter':
                        difficulty = np.random.choice(range(56,58),1)[0]
                        difflevel = 'medium'
                    elif task_name == 'orientation':
                        difficulty = np.random.choice(range(50,53),1)[0]
                        difflevel = 'medium'
                elif condition [trial] == 2 or condition[trial] == 5:
                    if task_name == 'color':
                        difficulty = np.random.choice(range(43,45),1)[0]
                        difflevel = 'hard'
                    elif task_name == 'shape':
                        difficulty = np.random.choice(range(48,50),1)[0]
                        difflevel = 'hard'
                    elif task_name == 'letter':
                        difficulty = np.random.choice(range(48,50),1)[0]
                        difflevel = 'hard'
                    elif task_name == 'orientation':
                        difficulty = np.random.choice(range(45,47),1)[0]
                        difflevel = 'hard'
                #Select relevant variables for this trial
                if condition[trial] < 3:
                    right = difficulty 
                    left = 80 - right
                    correct = 'right'
                elif condition[trial] >= 3:
                    left = difficulty 
                    right = 80 - left
                    correct = 'left'  
                           
                #create the actual stimuli
                if task_name == "color":
                    left_stim = vis.DotStim(win, units = 'norm', nDots = left, fieldSize = (0.5,0.9),fieldPos=(0,0), fieldShape = 'sqr', dotSize = 7, speed = 0, color = 'red')
                    right_stim = vis.DotStim(win, units = 'norm', nDots = right, fieldSize = (0.5,0.9),fieldPos=(0,0), fieldShape = 'sqr', dotSize = 7, speed = 0, color = 'blue')
                elif task_name == "letter":
                    left_stim = vis.TextStim(win,units="pix", text = 'A', color = 'white', height = 30)
                    right_stim = vis.TextStim(win,units="pix", text = 'B', color = 'white', height = 30) 
                    # shuffle the locations 
                    random.shuffle(coordinates[task_name])            
                elif task_name == "shape":
                    left_stim = vis.Rect(win, units = 'pix',lineColor = 'white', fillColor = 'black', size = 20)
                    right_stim = vis.Circle(win,units = 'pix', lineColor = 'white', fillColor = 'black', size = 20)    
                    # shuffle the locations 
                    random.shuffle(coordinates[task_name])
                elif task_name == "orientation":
                    #left_stim = vis.Line(win, start = (-10,0), end = (10, 0), units="pix", color = 'white') didn't work with version psychopy on eeg computer, had to switch the code 
                   # right_stim = vis.Line(win, start = (0,-10), end = (0, 10), units="pix", color = 'white')
                    left_stim = vis.TextStim(win,units="pix", text = '—', color = 'white', height = 20)
                    right_stim = vis.TextStim(win,units="pix", text = '|', color = 'white', height = 20) 

                    # shuffle the locations 
                    random.shuffle(coordinates[task_name])            
         
                ##Now that we have all, we start presenting stuff
                ##1. show a fixation cross
                rect.draw()
                resp_labels_left[task_name].draw()
                resp_labels_right[task_name].draw()
                fix.draw()
                if running == 'main':
                    cj_labels.draw()
                win.flip()
                sleep(time_cross)
                clock.reset()

                ##2. Show dots for 200 ms while querying a response without time limit
                resp=[];event.clearEvents();RT=0;clear_to_labels=0;
                rect.draw()
                resp_labels_left[task_name].draw()
                resp_labels_right[task_name].draw()
                if running == 'main':
                    cj_labels.draw()
                if task_name == "color":
                    left_stim.draw()
                    right_stim.draw()
                else:
                    for i in range(left + right): 
                        if i < left :
                            left_stim.pos = coordinates[task_name][i]
                            left_stim.draw()
                        else:
                            right_stim.pos = coordinates[task_name][i]
                            right_stim.draw()
                if trigger:   
                    sendTrigger(Trig_Begin_Target)
                win.flip()
        
                while(clear_to_labels<time_stim):
                    if len(resp) == 0:
                        resp  = event.getKeys(keyList = choice_keys)
                        RT = clock.getTime()
                    clear_to_labels=clock.getTime();
        
                ##3. Query a response
                while len(resp) == 0:
                    rect.draw()
                    resp_labels_left[task_name].draw()
                    resp_labels_right[task_name].draw()
                    if running == 'main':
                        cj_labels.draw()
                    win.flip()
                    resp  = event.getKeys(keyList = choice_keys)
                    RT = clock.getTime()

        
                if len(resp) > 1:
                    resp = [resp[0]]
                if trigger:
                    if resp[0] == choice_keys[0]:
                        sendTrigger(Trig_left_ERP)  #links
                    else:
                        sendTrigger(Trig_right_ERP) #rechts
                clock.reset()             
                
                ##4. Evaluate the response
                if correct=='left' and resp[0] == choice_keys[0]:
                    ACC=1
                elif correct=='right' and resp[0] == choice_keys[0]:
                    ACC=0
                elif correct=='left' and resp[0] == choice_keys[1]:
                    ACC=0
                elif correct=='right' and resp[0] == choice_keys[1]:
                    ACC=1
                accuracy[trial] = ACC
              
                ##Abord if escape
                if resp == ['escape']:
                    print('Participant pressed escape')
                    win.close()
                    core.quit()       
              
                #give feedback during training phase  
                score = -99
                if running == 'trainingphase':
                    if fb_name == 'positivefb' or fb_name == 'negativefb':
                        if trial == 17: 
                            event.clearEvents()
                            score = givefeedback(fb_name, block) 
                            if trigger:
                                sendTrigger(Trig_Begin_Feedback)
                    elif fb_name == 'easy' or fb_name == 'hard':
                        if ACC == 1:
                            rect.draw()
                            resp_labels_left[task_name].draw()
                            resp_labels_right[task_name].draw()
                            good.draw()
                        else:
                            rect.draw()
                            resp_labels_left[task_name].draw()
                            resp_labels_right[task_name].draw()
                            bad.draw()
                        win.flip()
                        sleep(time_fb)      
               
                #6. Ask for confidence about the choice after from the sixth block on
                if block >= nb_training_blocks:
                    clock.reset()   
                    rect.draw()
                    resp_labels_left[task_name].draw()
                    resp_labels_right[task_name].draw()
                    cj_labels.draw()
                    win.flip()
                    conf_press = event.waitKeys(keyList = cj_keys)
                    rtconf = clock.getTime()
                    RTconf = int(np.round(rtconf*1000))
                    if trigger: 
                        if conf_press[0] == cj_keys[0]:
                                sendTrigger(Trig_cj1)
                                event.clearEvents()
                                cj_label0 = vis.TextStim(win,text='Zeker fout      ', pos=(-0.35,0.8),height=.04, color = 'white')
                                cj_labels.draw()
                                cj_label0.draw()
                                rect.draw()
                                resp_labels_left[task_name].draw()
                                resp_labels_right[task_name].draw()
                                win.flip()
                                sleep(0.75)
                        elif conf_press[0] == cj_keys[1]:
                                sendTrigger(Trig_cj2)
                                event.clearEvents()
                                cj_label1 = vis.TextStim(win,text='Vermoed fout       ', pos=(-0.1925,0.8),height=.04, color = 'white')
                                cj_labels.draw()
                                cj_label1.draw()
                                rect.draw()
                                resp_labels_left[task_name].draw()
                                resp_labels_right[task_name].draw()
                                win.flip()
                                sleep(0.75)
                        elif conf_press[0] == cj_keys[2]:
                                sendTrigger(Trig_cj3)
                                event.clearEvents()
                                cj_label2 = vis.TextStim(win,text='Gok fout      ', pos=(-0.0475,0.8),height=.04, color = 'white')
                                cj_labels.draw()
                                cj_label2.draw()
                                rect.draw()
                                resp_labels_left[task_name].draw()
                                resp_labels_right[task_name].draw()
                                win.flip()
                                sleep(0.75)
                        elif conf_press[0] == cj_keys[3]:
                                sendTrigger(Trig_cj4)
                                event.clearEvents()
                                cj_label3 = vis.TextStim(win,text='Gok juist      ', pos=(0.0775,0.8),height=.04, color = 'white')
                                cj_labels.draw()
                                cj_label3.draw()
                                rect.draw()
                                resp_labels_left[task_name].draw()
                                resp_labels_right[task_name].draw()
                                win.flip()
                                sleep(0.75)
                        elif conf_press[0] == cj_keys[4]:
                                sendTrigger(Trig_cj5)
                                event.clearEvents()
                                cj_label4 = vis.TextStim(win,text='Vermoed juist      ', pos=(0.2275,0.8),height=.04, color = 'white')
                                cj_labels.draw()
                                cj_label4.draw()
                                rect.draw()
                                resp_labels_left[task_name].draw()
                                resp_labels_right[task_name].draw()
                                win.flip()
                                sleep(0.75)
                        elif conf_press[0] == cj_keys[5]:
                                sendTrigger(Trig_cj6)
                                event.clearEvents()
                                cj_label5 = vis.TextStim(win,text='Zeker juist      ', pos=(0.385,0.8),height=.04, color = 'white')
                                cj_labels.draw()
                                cj_label5.draw()
                                rect.draw()
                                resp_labels_left[task_name].draw()
                                resp_labels_right[task_name].draw()
                                win.flip()
                                sleep(0.75)
            #Convert conf_press into numeric value from 1 (sure error) tot 6
                    for temp in range(0,6):
                        if conf_press[0] == cj_keys[temp]:
                            cj = temp+1
                else:
                    conf_press = 'none'
                    cj = -99
                    RTconf = -99
                
                
                ##5. Store data of current trial
                thisExp.addData("task", task_name)
                thisExp.addData("withinblocktrial", trial)
                thisExp.addData("block", block)
                thisExp.addData("running", running)
                thisExp.addData("fb_name", fb_name)
                thisExp.addData("trialdifflevel", difflevel)
                thisExp.addData("dotsLeft", left)
                thisExp.addData("dotsRight", right)
                thisExp.addData("rt", RT)
                thisExp.addData("resp", resp)
                thisExp.addData("cor", ACC)
                thisExp.addData("cresp", correct)
                thisExp.addData("conf_press", conf_press)
                thisExp.addData("cj", cj)
                thisExp.addData("RTconf", RTconf)
                thisExp.addData("fbscore", score)
                # Proceed to next trial
                thisExp.nextEntry()
           
                   
            if running == "main":
                    accuracy = np.mean(accuracy)
                    pause= vis.TextStim(win,text="Einde van het block! Druk op C of N om verder te gaan.", pos=(0,0),height=.05)
                    pause.draw()
                    win.flip()
                    win.getMovieFrame()
                    win.saveMovieFrames("fb_good.png")
                    sleep(1)
                    event.waitKeys(keyList = 'space')
if trigger:
    serialport.close()
    

end = vis.TextStim(win,text='Einde van het experiment! Heel erg bedankt voor je deelname!!!!', pos=(0,0),height=.05)
end.draw();win.flip()
event.waitKeys(keyList = 'space')

##End of the experiment
win.flip()
core.quit()
