#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.5),
    on Thu Oct 31 11:07:23 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'sounddevice'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

from questplus import QuestPlus

####### QUESTPLUS INITIALIZATION ####################################################################################################################################################################################################

stim_domain = {'intensity': np.arange(0.01, 1, 0.01)}
param_domain = {
    'threshold': np.arange(0.01, 1, 0.01),
    'slope': np.arange(2, 5, 1),
    'lower_asymptote': 0.5, # Equal to chance
    'lapse_rate': np.arange(0, 0.05, 0.01) # Test 0:0.05 for adults, Consider 0:0.10 for children
}
outcome_domain = {'response': [1,0]}  # I'm going to flip this, to see if it fixes the way I intuitively think the algorithm should work; TDW 2025-01-22

# *THREE* QuestPlus staircases - one for each condition
qp_valid = QuestPlus(
    stim_domain=stim_domain,
    param_domain=param_domain,
    outcome_domain=outcome_domain,
    func='weibull',
    stim_scale='linear'
)

qp_invalid = QuestPlus(
    stim_domain=stim_domain,
    param_domain=param_domain,
    outcome_domain=outcome_domain,
    func='weibull',
    stim_scale='linear'
)

qp_neutral = QuestPlus(
    stim_domain=stim_domain,
    param_domain=param_domain,
    outcome_domain=outcome_domain,
    func='weibull',
    stim_scale='linear'
)

####### EXPERIMENT PARAMETERS ####################################################################################################################################################################################################

TRIAL_REPETITIONS = 16 # How many times to repeat each of the 12 unique trial types ( Total # trials = TRIAL_REPEITIONS * 12)
PRACTRIALS_REPETITIONS = 1 # Same as above, but for practice trials
CUE_SIZE = [.5, .5]
TARGET_SIZE = [1.5, 1.5]
FIXATION_SIZE = [.5, .5]
POSITION = np.array([5.0, 0.0]) # 5DVA eccentricity 
SPATIAL_FREQUENCY = 5

# Timing
ITI = 1.0 # fixation point between trials (s)
CUE_DURATION = 0.05 # (s)
ISI = 0.1 # fixation point between cue and target (s)
TARGET_DURATION = 1.0 # (s)
RESPONSE_DURATION = 1.0 # total response window is TARGET_DURATION + RESPONSE_DURATION (s)
TOTAL_TRIAL_DURATION = ITI + CUE_DURATION + ISI + TARGET_DURATION + RESPONSE_DURATION

####### PSYCHOPY SETUP ####################################################################################################################################################################################################

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.5'
expName = 'ViLoN_Posner_Cueing'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': '',
    'session': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1512, 982)
_loggingLevel = logging.getLevel('info')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/trentonwirth/GitHub/posner_attention/Experiment/ViLoN_Posner_Cueing.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.WARNING)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.WARNING)
    
    return logFile

def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # --- Initialize components for Routine "MainInstruc" ---
    mainInst = visual.TextStim(win=win, name='mainInst',
        text='''Welcome to the Line Grate Game!\n\n\n In this game, you will see line grates like these:\n\n\n\n
        Your job is to determine which way the lines are pointing!\n\nPress the left button if the lines are pointing up and down, 
        and the right button if the lines are pointing side to side.''',
        font='Arial',
        units='height', pos=(0, 0), draggable=False, height=0.04, wrapWidth=1700, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    instResp = keyboard.Keyboard(deviceName='instResp')
    Gabor_Inst1 = visual.GratingStim(
        win=win, name='Gabor',units='deg', 
        tex='sin', mask='gauss', anchor='center',
        ori=90, pos=[5,0], size=[4,4], sf=[SPATIAL_FREQUENCY], phase=0.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-2.0)
    Gabor_Inst2 = visual.GratingStim(
        win=win, name='Gabor',units='deg', 
        tex='sin', mask='gauss', anchor='center',
        ori=0.0, pos=[-5,0], size=[4,4], sf=[SPATIAL_FREQUENCY], phase=0.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-2.0)

    # --- Initialize components for Routines "practice and trial" ---
    Fixation_Point = visual.ShapeStim(
        win=win, name='Fixation_Point', vertices='cross',units='deg', 
        size=(FIXATION_SIZE[0], FIXATION_SIZE[1]),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=1.0, depth=0.0, interpolate=True)
    Left_Cue = visual.ShapeStim(
        win=win, name='Left_Cue',units='deg', 
        size=[CUE_SIZE[0], CUE_SIZE[1]], vertices='circle',
        ori=0.0, pos=[-POSITION[0], POSITION[1]], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=1.0, depth=-1.0, interpolate=True)
    Right_Cue = visual.ShapeStim(
        win=win, name='Right_Cue',units='deg', 
        size=[CUE_SIZE[0], CUE_SIZE[1]], vertices='circle',
        ori=0.0, pos=[POSITION[0], POSITION[1]], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=1.0, depth=-1.0, interpolate=True)
    Gabor = visual.GratingStim(
        win=win, name='Gabor',units='deg', 
        tex='sin', mask='gauss', anchor='center',
        ori=0.0, pos=[POSITION[0],POSITION[1]], size=[TARGET_SIZE[0], TARGET_SIZE[1]], sf=[SPATIAL_FREQUENCY], phase=0.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='add',
        texRes=128.0, interpolate=True)
    feedback = visual.TextStim(win=win, name='feedback',
        text="",
        font='Arial',
        units='height', pos=(0, 0), draggable=False, height=0.04, wrapWidth=1700, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')

    # --- Initialize components for Routine "End" ---
    endText = visual.TextStim(win=win, name='endText',
        text='Great job!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=1700, ori=0, 
        color='black', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    endKey = keyboard.Keyboard(deviceName='endKey')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
        # --- Prepare to start Routine "MainInstruc" ---
    # create an object to store info about Routine MainInstruc
    MainInstruc = data.Routine(
        name='MainInstruc',
        components=[mainInst, instResp, Gabor_Inst1, Gabor_Inst2],
    )
    MainInstruc.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instResp
    instResp.keys = []
    instResp.rt = []
    _instResp_allKeys = []
    # store start times for MainInstruc
    MainInstruc.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    MainInstruc.tStart = globalClock.getTime(format='float')
    MainInstruc.status = STARTED
    thisExp.addData('MainInstruc.started', MainInstruc.tStart)
    MainInstruc.maxDuration = None
    # keep track of which components have finished
    MainInstrucComponents = MainInstruc.components
    for thisComponent in MainInstruc.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "MainInstruc" ---
    MainInstruc.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *mainInst* updates
        
        # if mainInst is starting this frame...
        if mainInst.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mainInst.frameNStart = frameN  # exact frame index
            mainInst.tStart = t  # local t and not account for scr refresh
            mainInst.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mainInst, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'mainInst.started')
            # update status
            mainInst.status = STARTED
            mainInst.setAutoDraw(True)
        
        # if mainInst is active this frame...
        if mainInst.status == STARTED:
            # update params
            pass
        
        # *Gabor_Inst1* updates
        
        # if Gabor_Inst1 is starting this frame...
        if Gabor_Inst1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Gabor_Inst1.frameNStart = frameN  # exact frame index
            Gabor_Inst1.tStart = t  # local t and not account for scr refresh
            Gabor_Inst1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Gabor_Inst1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Gabor_Inst1.started')
            # update status
            Gabor_Inst1.status = STARTED
            Gabor_Inst1.setAutoDraw(True)
        
        # if Gabor_Inst1 is active this frame...
        if Gabor_Inst1.status == STARTED:
            # update params
            pass
        
        # *Gabor_Inst2* updates
        
        # if Gabor_Inst2 is starting this frame...
        if Gabor_Inst2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Gabor_Inst2.frameNStart = frameN  # exact frame index
            Gabor_Inst2.tStart = t  # local t and not account for scr refresh
            Gabor_Inst2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Gabor_Inst2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Gabor_Inst2.started')
            # update status
            Gabor_Inst2.status = STARTED
            Gabor_Inst2.setAutoDraw(True)
        
        # if Gabor_Inst2 is active this frame...
        if Gabor_Inst2.status == STARTED:
            # update params
            pass
        
        # *instResp* updates
        waitOnFlip = False
        
        # if instResp is starting this frame...
        if instResp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instResp.frameNStart = frameN  # exact frame index
            instResp.tStart = t  # local t and not account for scr refresh
            instResp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instResp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instResp.started')
            # update status
            instResp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instResp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instResp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instResp.status == STARTED and not waitOnFlip:
            theseKeys = instResp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instResp_allKeys.extend(theseKeys)
            if len(_instResp_allKeys):
                instResp.keys = _instResp_allKeys[-1].name  # just the last key pressed
                instResp.rt = _instResp_allKeys[-1].rt
                instResp.duration = _instResp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            MainInstruc.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in MainInstruc.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "MainInstruc" ---
    for thisComponent in MainInstruc.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for MainInstruc
    MainInstruc.tStop = globalClock.getTime(format='float')
    MainInstruc.tStopRefresh = tThisFlipGlobal
    thisExp.addData('MainInstruc.stopped', MainInstruc.tStop)
    # check responses
    if instResp.keys in ['', [], None]:  # No response was made
        instResp.keys = None
    thisExp.addData('instResp.keys',instResp.keys)
    if instResp.keys != None:  # we had a response
        thisExp.addData('instResp.rt', instResp.rt)
        thisExp.addData('instResp.duration', instResp.duration)
    thisExp.nextEntry()
    # the Routine "MainInstruc" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()

####### FUNCTIONS #################################################################################################################################################################################################### 

    # Determine the opacity (location) of the cue based on cue condition
    def get_cue_opacity(cue_condition, gabor_position):
        if cue_condition == 'Neutral': # show both cues
            return 1.0, 1.0 
        elif cue_condition == 'Valid': # show cue in same position as target gabor
            if gabor_position == -1:
                return 1.0, 0.0 
            else:
                return 0.0, 1.0
        elif cue_condition == 'Invalid': # show cue in opposite position as target gabor
            if gabor_position == 1:
                return 1.0, 0.0
            else:
                return 0.0, 1.0

    # Get the list of trials from the TrialHandler
    def get_trials(handler):
        trial_sequence = handler.sequenceIndices # 2D array of the order in which trialList indices will be presented
        trial_indices = trial_sequence.T.flatten().tolist() # Transpose and flatten the array to get a list of trial indices
        trials = [handler.trialList[i] for i in trial_indices]  # Get the actual trial data from the handler
        return trials

    # Check that there are no more than 3 consecutive trials with the same cue condition
    def consecutive_check(all_trials, max_repeats=3):
        consecutive_count = 1
        for i in range(1, len(all_trials)):
            if all_trials[i]['cue_condition'] == all_trials[i - 1]['cue_condition']:
                consecutive_count += 1
                if consecutive_count > max_repeats:
                    return False  # Invalid trial list
            else:
                consecutive_count = 1
        return True  # Valid trial list
    

####### PRACTICE TRIAL SETUP #################################################################################################################################################################################################### 
    
    # Create a dictionary of 12 unique trial types used to create practice and experiment trial lists
    trialList = data.createFactorialTrialList({
                'orientation': [0, 90], # 0 - vertical; 90 - horizontal
                'gabor_position': [-1, 1], # -1 = Left, 1 = Right
                'cue_condition': ['Neutral', 'Invalid','Valid']  
                }) 
    
    # While accuracy is below threshold, repeat practice block (2 max repetitions)
    repeat_count = 0 
    repeat_practice = True
    while repeat_practice:
        repeat_count += 1
        total_correct = 0

        # Create initial list of practice trials
        Prac_Rep = data.TrialHandler(nReps=PRACTRIALS_REPETITIONS, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=trialList,  # Using our conditions defined above
            seed=None, name='Prac_Rep')

        # Check whether list of trials has no more than three of the same cue condition in a row, reshuffle until it meets this criteria
        while consecutive_check(get_trials(Prac_Rep)) == False:
            Prac_Rep = data.TrialHandler(nReps=PRACTRIALS_REPETITIONS, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=trialList,
            seed=None, name='Prac_Rep')

        # print('Practice sequence:', Prac_Rep.sequenceIndices)
        # print('Practice trial list:', get_trials(Prac_Rep))
        
        thisExp.addLoop(Prac_Rep)  # add the practice block to the experiment
        thisPrac_Rep = Prac_Rep.trialList[0] 
    
        # abbreviate parameter names if possible (e.g. rgb = thisPrac_Rep.rgb)
        if thisPrac_Rep != None:
            for paramName in thisPrac_Rep:
                globals()[paramName] = thisPrac_Rep[paramName]

        # Run practice trials
        for thisPrac_Rep in Prac_Rep:

            # Set trial visuals
            Gabor.pos = np.array([POSITION[0] * thisPrac_Rep['gabor_position'], POSITION[1]])
            Gabor.ori = thisPrac_Rep['orientation']
            Left_Cue.opacity, Right_Cue.opacity = get_cue_opacity(thisPrac_Rep['cue_condition'], thisPrac_Rep['gabor_position'])

            # In practice trials, randomly set gabor contrast to 0.1, 0.5, or 1.0
            pract_contrasts = [0.1, 0.5, 1] * 4
            shuffle(pract_contrasts)
            intensity = pract_contrasts.pop(0)
            Gabor.contrast = intensity

            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )

####### PSYCHOPY RUNS PRACTICE TRIALS ####################################################################################################################################################################################################
  
                    # --- Prepare to start Routine "practice" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('practice.started', globalClock.getTime(format='float'))
            # create starting attributes for key_resp
            key_resp.keys = []
            key_resp.rt = []
            _key_resp_allKeys = []
            # keep track of which components have finished
            pracComponents = [Fixation_Point, Left_Cue, Right_Cue, Gabor, key_resp]
            for thisComponent in pracComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "practice" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *Fixation_Point* updates
                
                # if Fixation_Point is starting this frame...
                if Fixation_Point.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    Fixation_Point.color = 'white'
                    # keep track of start time/frame for later
                    Fixation_Point.frameNStart = frameN  # exact frame index
                    Fixation_Point.tStart = t  # local t and not account for scr refresh
                    Fixation_Point.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Fixation_Point, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Fixation_Point.started')
                    # update status
                    Fixation_Point.status = STARTED
                    Fixation_Point.setAutoDraw(True)
                
                # if Fixation_Point is active this frame...
                if Fixation_Point.status == STARTED:
                    # update params
                    pass
                    
                # if Fixation_Point is stopping this frame...
                if Fixation_Point.status == STARTED:
                    # Fixation point turns green for last 100 ms of ITI
                    if tThisFlip >= 0.9-frameTolerance and tThisFlip < ITI-frameTolerance:
                        Fixation_Point.color = 'green'
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Fixation_Point.tStartRefresh + TOTAL_TRIAL_DURATION-frameTolerance:
                        # keep track of stop time/frame for later
                        Fixation_Point.tStop = t  # not accounting for scr refresh
                        Fixation_Point.tStopRefresh = tThisFlipGlobal  # on global time
                        Fixation_Point.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Fixation_Point.stopped')
                        # update status
                        Fixation_Point.status = FINISHED
                        Fixation_Point.setAutoDraw(False)

                # Fixation point turns white when cue appears (left and right cues appear at the same time, so only one needs to be checked)
                if Left_Cue.status == STARTED: 
                    Fixation_Point.color = 'white'  

                # FP to blue when the target disappears
                if Gabor.status == FINISHED:
                    Fixation_Point.color = 'blue'   

                # *Left_Cue* updates
                
                # if Left_Cue is starting this frame...
                if Left_Cue.status == NOT_STARTED and tThisFlip >= ITI-frameTolerance:
                    # keep track of start time/frame for later
                    Left_Cue.frameNStart = frameN  # exact frame index
                    Left_Cue.tStart = t  # local t and not account for scr refresh
                    Left_Cue.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Left_Cue, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Left_Cue.started')
                    # update status
                    Left_Cue.status = STARTED
                    Left_Cue.setAutoDraw(True)
                
                # if Left_Cue is active this frame...
                if Left_Cue.status == STARTED:
                    # update params
                    pass
                
                # if Left_Cue is stopping this frame...
                if Left_Cue.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Left_Cue.tStartRefresh + CUE_DURATION-frameTolerance:
                        # keep track of stop time/frame for later
                        Left_Cue.tStop = t  # not accounting for scr refresh
                        Left_Cue.tStopRefresh = tThisFlipGlobal  # on global time
                        Left_Cue.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Left_Cue.stopped')
                        # update status
                        Left_Cue.status = FINISHED
                        Left_Cue.setAutoDraw(False)
                
                # *Right_Cue* updates
                
                # if Right_Cue is starting this frame...
                if Right_Cue.status == NOT_STARTED and tThisFlip >= ITI-frameTolerance:
                    # keep track of start time/frame for later
                    Right_Cue.frameNStart = frameN  # exact frame index
                    Right_Cue.tStart = t  # local t and not account for scr refresh
                    Right_Cue.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Right_Cue, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Right_Cue.started')
                    # update status
                    Right_Cue.status = STARTED
                    Right_Cue.setAutoDraw(True)
                
                # if Right_Cue is active this frame...
                if Right_Cue.status == STARTED:
                    # update params
                    pass
                
                # if Right_Cue is stopping this frame...
                if Right_Cue.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Right_Cue.tStartRefresh + CUE_DURATION-frameTolerance:
                        # keep track of stop time/frame for later
                        Right_Cue.tStop = t  # not accounting for scr refresh
                        Right_Cue.tStopRefresh = tThisFlipGlobal  # on global time
                        Right_Cue.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Right_Cue.stopped')
                        # update status
                        Right_Cue.status = FINISHED
                        Right_Cue.setAutoDraw(False)
                
                # *Gabor* updates
                
                # if Gabor is starting this frame...
                if Gabor.status == NOT_STARTED and tThisFlip >= (ITI+CUE_DURATION+ISI)-frameTolerance:
                    # keep track of start time/frame for later
                    Gabor.frameNStart = frameN  # exact frame index
                    Gabor.tStart = t  # local t and not account for scr refresh
                    Gabor.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Gabor, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Gabor.started')
                    # update status
                    Gabor.status = STARTED
                    Gabor.setAutoDraw(True)
                
                # if Gabor is active this frame...
                if Gabor.status == STARTED:
                    # update params
                    pass
                
                # if Gabor is stopping this frame...
                if Gabor.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Gabor.tStartRefresh + TARGET_DURATION-frameTolerance:
                        # keep track of stop time/frame for later
                        Gabor.tStop = t  # not accounting for scr refresh
                        Gabor.tStopRefresh = tThisFlipGlobal  # on global time
                        Gabor.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Gabor.stopped')
                        # update status
                        Gabor.status = FINISHED
                        Gabor.setAutoDraw(False)
                
                # *key_resp* updates
                waitOnFlip = False
                
                # if key_resp is starting this frame...
                if key_resp.status == NOT_STARTED and tThisFlip >= (ITI+CUE_DURATION+ISI)-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp.frameNStart = frameN  # exact frame index
                    key_resp.tStart = t  # local t and not account for scr refresh
                    key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.started')
                    # update status
                    key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    
                if key_resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp.tStartRefresh + (TARGET_DURATION+RESPONSE_DURATION)-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp.tStop = t  # not accounting for scr refresh
                        key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                        key_resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp.stopped')
                        # update status
                        key_resp.status = FINISHED
                        key_resp.status = FINISHED    
                        
                if key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp.getKeys(keyList=['1','2'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_allKeys.extend(theseKeys)
                    if len(_key_resp_allKeys):
                        key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                        key_resp.rt = _key_resp_allKeys[-1].rt
                        key_resp.duration = _key_resp_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                    
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in pracComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()

            # End of Psychopy builder code that runs the practice block
            # --- Ending Routine "practice" ---
            for thisComponent in pracComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            
            # Save trial visuals to the data file
            thisExp.addData('practice.stopped', globalClock.getTime(format='float'))
            # Add intensity to the data file
            thisExp.addData('Gabor.intensity', intensity)
            # Add position to the data file
            thisExp.addData('Gabor.pos', Gabor.pos)
            # Add orientation to the data file
            thisExp.addData('Gabor.ori', Gabor.ori)
            # Add opacity of each cue to the data file
            thisExp.addData('Left_Cue.opacity', Left_Cue.opacity)
            thisExp.addData('Right_Cue.opacity', Right_Cue.opacity)

####### PRACTICE TRIALS RESPONSE CHECK #################################################################################################################################################################################################### 

            # check responses and give feedback for each trial
            if key_resp.keys in ['', [], None]:  # No response was made
                key_resp.keys = None
                response = None
                feedback.text = "Try Again!"
                feedback.draw()
                win.flip()
                core.wait(1)
            else:
                response = 1 if (
                    (key_resp.keys == '1' and thisPrac_Rep['orientation'] == 0) or 
                    (key_resp.keys == '2' and thisPrac_Rep['orientation'] == 90)
                    ) else 0
                if response == 1:
                    total_correct += 1
                    feedback.text = "Correct!"
                    feedback.draw()
                    win.flip()
                    core.wait(1)
                elif response == 0:
                    feedback.text = "Try Again!"
                    feedback.draw()
                    win.flip()
                    core.wait(1)         

            # Save response, accuracy, RT, and duration to data file and print response
            Prac_Rep.addData('key_resp.keys',key_resp.keys)
            Prac_Rep.addData('Accuracy', response)
    
            if key_resp.keys != None: 
                Prac_Rep.addData('key_resp.rt', key_resp.rt)
                Prac_Rep.addData('key_resp.duration', key_resp.duration)
            
            print("Response:", response)

            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                thisSession.sendExperimentData()
        # End of practice block

        # Calculate accuracy on practice trials, determine whether to continue to experiment block, repeat practice, or end the experiment
        percent_correct = (total_correct / (len(Prac_Rep.trialList)*PRACTRIALS_REPETITIONS)) * 100
        print(f"Percent correct: {percent_correct}")

        if percent_correct >= 75: # Continue to experiment block
            start_text = visual.TextStim(win, text="Great Job!\n\nAre you ready to play the real game? ", color="black", units='pix', height=40, wrapWidth = 1700)
            print("Practice block passed! Starting experiment trials.")
            start_text.draw()
            win.flip()
            event.waitKeys(keyList=['space'])
            repeat_practice = False
            core.wait(1)
        else:
            if repeat_count < 2: # Repeat practice block
                retry_text = visual.TextStim(win, text="Let's try the practice again!", color="black", units='pix', height=40, wrapWidth=1700)
                retry_text.draw()
                win.flip()
                event.waitKeys(keyList=['space'])
                core.wait(1)
            else: # End experiment
                quit_text = visual.TextStim(win, text="You finished the game!", color="black", units='pix', height=40, wrapWidth=1700)
                print("Participant did not pass practice trials. Ending experiment.")
                quit_text.draw()
                win.flip()
                event.waitKeys(keyList=['space'])
                core.quit()

####### EXPERIMENT TRIAL SETUP #################################################################################################################################################################################################### 

    # Create initial list of experiment trials
    Trial_Rep = data.TrialHandler(nReps=TRIAL_REPETITIONS, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=trialList,  # Using our conditions defined above
        seed=None, name='Trial_Rep')

    while consecutive_check(get_trials(Trial_Rep)) == False:
        print("Reshuffling experiment trials to avoid consecutive cue conditions.")
        Trial_Rep = data.TrialHandler(nReps=TRIAL_REPETITIONS, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=trialList,  # Using our conditions defined above
        seed=None, name='Trial_Rep')

    # print("Experiment Sequence Indices:", Trial_Rep.sequenceIndices)
    print("Experiment Trial List:", get_trials(Trial_Rep))

    thisExp.addLoop(Trial_Rep)  # add experiment block to the experiment
    thisTrial_Rep = Trial_Rep.trialList[0] 
    no_resp_trials = [] # inititate list of trials that participant did not respond to

    # abbreviate parameter names if possible (e.g. rgb = thisPrac_Rep.rgb)
    if thisTrial_Rep != None:
        for paramName in thisTrial_Rep:
            globals()[paramName] = thisTrial_Rep[paramName]
    
    # Run each experiment trial
    for thisTrial_Rep in Trial_Rep:        
        
        # Set trial visuals
        Gabor.pos = np.array([POSITION[0] * thisTrial_Rep['gabor_position'], POSITION[1]])
        Left_Cue.opacity, Right_Cue.opacity = get_cue_opacity(
            cue_condition=thisTrial_Rep['cue_condition'],
            gabor_position=thisTrial_Rep['gabor_position']
        )

#        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        
        # Set Gabor orientation and assign trial to corresponding QP algorithm
        if thisTrial_Rep['orientation'] == 0 and thisTrial_Rep['cue_condition'] == 'Valid':
            Gabor.ori = 0
            current_qp = qp_valid

        elif thisTrial_Rep['orientation'] == 0 and not thisTrial_Rep['cue_condition'] == 'Invalid':
            Gabor.ori = 0
            current_qp = qp_invalid

        elif thisTrial_Rep['orientation'] == 90 and thisTrial_Rep['cue_condition'] == 'Valid':
            Gabor.ori = 90
            current_qp = qp_valid

        elif thisTrial_Rep['orientation'] == 90 and not thisTrial_Rep['cue_condition'] == 'Invalid':
            Gabor.ori = 90
            current_qp = qp_invalid
            
        elif thisTrial_Rep['orientation'] == 0 and thisTrial_Rep['cue_condition'] == 'Neutral':
            Gabor.ori = 0
            current_qp = qp_neutral

        elif thisTrial_Rep['orientation'] == 90 and not thisTrial_Rep['cue_condition'] == 'Neutral':
            Gabor.ori = 90
            current_qp = qp_neutral    
        
        # Save threshold, slope, and lapse rate from QP to data file
        threshold = current_qp.param_estimate['threshold']
        slope = current_qp.param_estimate['slope']
        lapse_rate = current_qp.param_estimate['lapse_rate']

        # Get next intensity from current staircase
        next_stim = current_qp.next_stim
        intensity = next_stim['intensity']
        print(f"Orientation: {thisTrial_Rep['orientation']}, Next Intensity: {intensity}, Cue condition: {thisTrial_Rep['cue_condition']}")
        
        # Update Gabor contrast
        Gabor.contrast = intensity
        
####### PSYCHOPY RUNS EXPERIMENT TRIALS ####################################################################################################################################################################################################

        # --- Prepare to start Routine "trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial.started', globalClock.getTime(format='float'))
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # keep track of which components have finished
        trialComponents = [Fixation_Point, Left_Cue, Right_Cue, Gabor, key_resp]
        for thisComponent in trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Fixation_Point* updates
            
            #FIXATION CROSS REMAINS ON THE SCREEN THROUGHOUT TRIAL (02/24/2025 VBG)
            # if Fixation_Point is starting this frame...
            if Fixation_Point.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                Fixation_Point.color = 'white'
                # keep track of start time/frame for later
                Fixation_Point.frameNStart = frameN  # exact frame index
                Fixation_Point.tStart = t  # local t and not account for scr refresh
                Fixation_Point.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Fixation_Point, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Fixation_Point.started')
                # update status
                Fixation_Point.status = STARTED
                Fixation_Point.setAutoDraw(True)
            
            # if Fixation_Point is active this frame...
            if Fixation_Point.status == STARTED:
                # update params
                pass
                
            # if Fixation_Point is stopping this frame...
            if Fixation_Point.status == STARTED:
                # Fixation point turns green for last 100 ms of ITI
                if tThisFlip >= 0.9-frameTolerance and tThisFlip < ITI-frameTolerance:
                    Fixation_Point.color = 'green'
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Fixation_Point.tStartRefresh + TOTAL_TRIAL_DURATION-frameTolerance:
                    # keep track of stop time/frame for later
                    Fixation_Point.tStop = t  # not accounting for scr refresh
                    Fixation_Point.tStopRefresh = tThisFlipGlobal  # on global time
                    Fixation_Point.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Fixation_Point.stopped')
                    # update status
                    Fixation_Point.status = FINISHED
                    Fixation_Point.setAutoDraw(False)
         
            # Fixation point turns white when cue appears (left and right cues appear at the same time, so only one needs to be checked)
            if Left_Cue.status == STARTED: 
                Fixation_Point.color = 'white'  

            # FP to blue when the target disappears
            if Gabor.status == FINISHED:
                Fixation_Point.color = 'blue' 
            
            # *Left_Cue* updates
            
            # if Left_Cue is starting this frame...
            if Left_Cue.status == NOT_STARTED and tThisFlip >= ITI-frameTolerance:
                # keep track of start time/frame for later
                Left_Cue.frameNStart = frameN  # exact frame index
                Left_Cue.tStart = t  # local t and not account for scr refresh
                Left_Cue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Left_Cue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Left_Cue.started')
                # update status
                Left_Cue.status = STARTED
                Left_Cue.setAutoDraw(True)
            
            # if Left_Cue is active this frame...
            if Left_Cue.status == STARTED:
                # update params
                pass
            
            # if Left_Cue is stopping this frame...
            if Left_Cue.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Left_Cue.tStartRefresh + CUE_DURATION-frameTolerance:
                    # keep track of stop time/frame for later
                    Left_Cue.tStop = t  # not accounting for scr refresh
                    Left_Cue.tStopRefresh = tThisFlipGlobal  # on global time
                    Left_Cue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Left_Cue.stopped')
                    # update status
                    Left_Cue.status = FINISHED
                    Left_Cue.setAutoDraw(False)
            
            # *Right_Cue* updates
            
            # if Right_Cue is starting this frame...
            if Right_Cue.status == NOT_STARTED and tThisFlip >= ITI-frameTolerance:
                # keep track of start time/frame for later
                Right_Cue.frameNStart = frameN  # exact frame index
                Right_Cue.tStart = t  # local t and not account for scr refresh
                Right_Cue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Right_Cue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Right_Cue.started')
                # update status
                Right_Cue.status = STARTED
                Right_Cue.setAutoDraw(True)
            
            # if Right_Cue is active this frame...
            if Right_Cue.status == STARTED:
                # update params
                pass
            
            # if Right_Cue is stopping this frame...
            if Right_Cue.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Right_Cue.tStartRefresh + CUE_DURATION-frameTolerance:
                    # keep track of stop time/frame for later
                    Right_Cue.tStop = t  # not accounting for scr refresh
                    Right_Cue.tStopRefresh = tThisFlipGlobal  # on global time
                    Right_Cue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Right_Cue.stopped')
                    # update status
                    Right_Cue.status = FINISHED
                    Right_Cue.setAutoDraw(False)
            
            # *Gabor* updates
            
            # if Gabor is starting this frame...
            if Gabor.status == NOT_STARTED and tThisFlip >= (ITI+CUE_DURATION+ISI)-frameTolerance:
                # keep track of start time/frame for later
                Gabor.frameNStart = frameN  # exact frame index
                Gabor.tStart = t  # local t and not account for scr refresh
                Gabor.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Gabor, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Gabor.started')
                # update status
                Gabor.status = STARTED
                Gabor.setAutoDraw(True)
            
            # if Gabor is active this frame...
            if Gabor.status == STARTED:
                # update params
                pass
            
            # if Gabor is stopping this frame...
            if Gabor.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Gabor.tStartRefresh + TARGET_DURATION-frameTolerance:
                    # keep track of stop time/frame for later
                    Gabor.tStop = t  # not accounting for scr refresh
                    Gabor.tStopRefresh = tThisFlipGlobal  # on global time
                    Gabor.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Gabor.stopped')
                    # update status
                    Gabor.status = FINISHED
                    Gabor.setAutoDraw(False)
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= (ITI+CUE_DURATION+ISI)-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
            if key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp.tStartRefresh + (TARGET_DURATION+RESPONSE_DURATION)-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp.tStop = t  # not accounting for scr refresh
                    key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.stopped')
                    # update status
                    key_resp.status = FINISHED
                    key_resp.status = FINISHED     
                    
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['1','2'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
                
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        
        thisExp.addData('trial.stopped', globalClock.getTime(format='float'))
        # Add intensity to the data file
        thisExp.addData('Gabor.intensity', intensity)
        # Add position to the data file
        thisExp.addData('Gabor.pos', Gabor.pos)
        # Add orientation to the data file
        thisExp.addData('Gabor.ori', Gabor.ori)
        # Add threshold, slope, and lapse rate to the data file
        thisExp.addData('QP_Threshold', threshold)
        thisExp.addData('QP_Slope', slope)
        thisExp.addData('QP_Lapse', lapse_rate)
        # Add opacity of each cue to the data file
        thisExp.addData('Left_Cue.opacity', Left_Cue.opacity)
        thisExp.addData('Right_Cue.opacity', Right_Cue.opacity)

####### EXPERIMENT TRIALS RESPONSE CHECK #################################################################################################################################################################################################### 

        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
            response = None
            no_resp_trials.append(thisTrial_Rep) # Add trial to list of trials with no response
        else:
            # Update QuestPlus with trial accuracy; QP is not updated if no response is made
            response = 1 if (
                (key_resp.keys == '1' and thisTrial_Rep['orientation'] == 0) or 
                (key_resp.keys == '2' and thisTrial_Rep['orientation'] == 90)
                ) else 0 
            current_qp.update(stim={'intensity': intensity}, outcome={'response': response})
        
        # Save response key, accuracy, RT, and duration to data file
        Trial_Rep.addData('key_resp.keys',key_resp.keys)
        Trial_Rep.addData('Accuracy', response)
        if key_resp.keys != None: 
            Trial_Rep.addData('key_resp.rt', key_resp.rt)
            Trial_Rep.addData('key_resp.duration', key_resp.duration)
        
        print("Response:", response)

        # the Routine "trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()

        # Participant can take a break every 32 trials
        if (Trial_Rep.thisN+1) % 32 == 0 and Trial_Rep.thisN != 191: 
            break_text = visual.TextStim(win, text="Take a quick break!", color="black", units='pix', height=40, wrapWidth=1700)
            break_text.draw()
            win.flip()
            event.waitKeys(keyList=['space'])
            core.wait(1) 

    # Completed all experiment trials
    # print('Trials with no response:', no_resp_trials)

####### REPEATING NO RESPONSE EXPERIMENT TRIALS ####################################################################################################################################################################################################    

    if no_resp_trials:
        print(f"Re-running {len(no_resp_trials)} trials with no response...")
        for trial in no_resp_trials:
            # Set up trial visuals
            Gabor.pos = np.array([POSITION[0] * trial['gabor_position'], POSITION[1]])
            Gabor.ori = trial['orientation']
            Left_Cue.opacity, Right_Cue.opacity = get_cue_opacity(
                        cue_condition=thisTrial_Rep['cue_condition'],
                        gabor_position=thisTrial_Rep['gabor_position']
                    )

            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            
            # Set Gabor orientation and assign trial to corresponding QP algorithm
            if thisTrial_Rep['orientation'] == 0 and thisTrial_Rep['cue_condition'] == 'Valid':
                Gabor.ori = 0
                current_qp = qp_valid

            elif thisTrial_Rep['orientation'] == 0 and not thisTrial_Rep['cue_condition'] == 'Invalid':
                Gabor.ori = 0
                current_qp = qp_invalid

            elif thisTrial_Rep['orientation'] == 90 and thisTrial_Rep['cue_condition'] == 'Valid':
                Gabor.ori = 90
                current_qp = qp_valid

            elif thisTrial_Rep['orientation'] == 90 and not thisTrial_Rep['cue_condition'] == 'Invalid':
                Gabor.ori = 90
                current_qp = qp_invalid
                
            elif thisTrial_Rep['orientation'] == 0 and thisTrial_Rep['cue_condition'] == 'Neutral':
                Gabor.ori = 0
                current_qp = qp_neutral

            elif thisTrial_Rep['orientation'] == 90 and not thisTrial_Rep['cue_condition'] == 'Neutral':
                Gabor.ori = 90
                current_qp = qp_neutral    
            
            # Save threshold, slope, and lapse rate from QP to data file
            threshold = current_qp.param_estimate['threshold']
            slope = current_qp.param_estimate['slope']
            lapse_rate = current_qp.param_estimate['lapse_rate']

            # Get next intensity from current staircase
            next_stim = current_qp.next_stim
            intensity = next_stim['intensity']
            print(f"Orientation: {thisTrial_Rep['orientation']}, Next Intensity: {intensity}, Cue condition: {thisTrial_Rep['cue_condition']}")
            
            # Update Gabor contrast
            Gabor.contrast = intensity

####### PSYCHOPY RUNS NO RESPONSE TRIALS ####################################################################################################################################################################################################

            # --- Prepare to start Routine "trial" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trial.started', globalClock.getTime(format='float'))
            # create starting attributes for key_resp
            key_resp.keys = []
            key_resp.rt = []
            _key_resp_allKeys = []
            # keep track of which components have finished
            trialComponents = [Fixation_Point, Left_Cue, Right_Cue, Gabor, key_resp]
            for thisComponent in trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *Fixation_Point* updates
                
                #FIXATION CROSS REMAINS ON THE SCREEN THROUGHOUT TRIAL (02/24/2025 VBG)
                # if Fixation_Point is starting this frame...
                if Fixation_Point.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    Fixation_Point.color = 'white'
                    # keep track of start time/frame for later
                    Fixation_Point.frameNStart = frameN  # exact frame index
                    Fixation_Point.tStart = t  # local t and not account for scr refresh
                    Fixation_Point.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Fixation_Point, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Fixation_Point.started')
                    # update status
                    Fixation_Point.status = STARTED
                    Fixation_Point.setAutoDraw(True)
                
                # if Fixation_Point is active this frame...
                if Fixation_Point.status == STARTED:
                    # update params
                    pass
                    
                # if Fixation_Point is stopping this frame...
                if Fixation_Point.status == STARTED:
                    # Fixation point turns green for last 100 ms of ITI
                    if tThisFlip >= 0.9-frameTolerance and tThisFlip < ITI-frameTolerance:
                        Fixation_Point.color = 'green'
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Fixation_Point.tStartRefresh + TOTAL_TRIAL_DURATION-frameTolerance:
                        # keep track of stop time/frame for later
                        Fixation_Point.tStop = t  # not accounting for scr refresh
                        Fixation_Point.tStopRefresh = tThisFlipGlobal  # on global time
                        Fixation_Point.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Fixation_Point.stopped')
                        # update status
                        Fixation_Point.status = FINISHED
                        Fixation_Point.setAutoDraw(False)
            
                # Fixation point turns white when cue appears (left and right cues appear at the same time, so only one needs to be checked)
                if Left_Cue.status == STARTED: 
                    Fixation_Point.color = 'white'  

                # FP to blue when the target disappears
                if Gabor.status == FINISHED:
                    Fixation_Point.color = 'blue' 
                
                # *Left_Cue* updates
                
                # if Left_Cue is starting this frame...
                if Left_Cue.status == NOT_STARTED and tThisFlip >= ITI-frameTolerance:
                    # keep track of start time/frame for later
                    Left_Cue.frameNStart = frameN  # exact frame index
                    Left_Cue.tStart = t  # local t and not account for scr refresh
                    Left_Cue.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Left_Cue, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Left_Cue.started')
                    # update status
                    Left_Cue.status = STARTED
                    Left_Cue.setAutoDraw(True)
                
                # if Left_Cue is active this frame...
                if Left_Cue.status == STARTED:
                    # update params
                    pass
                
                # if Left_Cue is stopping this frame...
                if Left_Cue.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Left_Cue.tStartRefresh + CUE_DURATION-frameTolerance:
                        # keep track of stop time/frame for later
                        Left_Cue.tStop = t  # not accounting for scr refresh
                        Left_Cue.tStopRefresh = tThisFlipGlobal  # on global time
                        Left_Cue.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Left_Cue.stopped')
                        # update status
                        Left_Cue.status = FINISHED
                        Left_Cue.setAutoDraw(False)
                
                # *Right_Cue* updates
                
                # if Right_Cue is starting this frame...
                if Right_Cue.status == NOT_STARTED and tThisFlip >= ITI-frameTolerance:
                    # keep track of start time/frame for later
                    Right_Cue.frameNStart = frameN  # exact frame index
                    Right_Cue.tStart = t  # local t and not account for scr refresh
                    Right_Cue.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Right_Cue, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Right_Cue.started')
                    # update status
                    Right_Cue.status = STARTED
                    Right_Cue.setAutoDraw(True)
                
                # if Right_Cue is active this frame...
                if Right_Cue.status == STARTED:
                    # update params
                    pass
                
                # if Right_Cue is stopping this frame...
                if Right_Cue.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Right_Cue.tStartRefresh + CUE_DURATION-frameTolerance:
                        # keep track of stop time/frame for later
                        Right_Cue.tStop = t  # not accounting for scr refresh
                        Right_Cue.tStopRefresh = tThisFlipGlobal  # on global time
                        Right_Cue.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Right_Cue.stopped')
                        # update status
                        Right_Cue.status = FINISHED
                        Right_Cue.setAutoDraw(False)
                
                # *Gabor* updates
                
                # if Gabor is starting this frame...
                if Gabor.status == NOT_STARTED and tThisFlip >= (ITI+CUE_DURATION+ISI)-frameTolerance:
                    # keep track of start time/frame for later
                    Gabor.frameNStart = frameN  # exact frame index
                    Gabor.tStart = t  # local t and not account for scr refresh
                    Gabor.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Gabor, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Gabor.started')
                    # update status
                    Gabor.status = STARTED
                    Gabor.setAutoDraw(True)
                
                # if Gabor is active this frame...
                if Gabor.status == STARTED:
                    # update params
                    pass
                
                # if Gabor is stopping this frame...
                if Gabor.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Gabor.tStartRefresh + TARGET_DURATION-frameTolerance:
                        # keep track of stop time/frame for later
                        Gabor.tStop = t  # not accounting for scr refresh
                        Gabor.tStopRefresh = tThisFlipGlobal  # on global time
                        Gabor.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Gabor.stopped')
                        # update status
                        Gabor.status = FINISHED
                        Gabor.setAutoDraw(False)
                
                # *key_resp* updates
                waitOnFlip = False
                
                # if key_resp is starting this frame...
                if key_resp.status == NOT_STARTED and tThisFlip >= (ITI+CUE_DURATION+ISI)-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp.frameNStart = frameN  # exact frame index
                    key_resp.tStart = t  # local t and not account for scr refresh
                    key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.started')
                    # update status
                    key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    
                if key_resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp.tStartRefresh + (TARGET_DURATION+RESPONSE_DURATION)-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp.tStop = t  # not accounting for scr refresh
                        key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                        key_resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp.stopped')
                        # update status
                        key_resp.status = FINISHED
                        key_resp.status = FINISHED     
                        
                if key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp.getKeys(keyList=['1','2'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_allKeys.extend(theseKeys)
                    if len(_key_resp_allKeys):
                        key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                        key_resp.rt = _key_resp_allKeys[-1].rt
                        key_resp.duration = _key_resp_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                    
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            
            thisExp.addData('trial.stopped', globalClock.getTime(format='float'))
            # Add intensity to the data file
            thisExp.addData('Gabor.intensity', intensity)
            # Add position to the data file
            thisExp.addData('Gabor.pos', Gabor.pos)
            # Add orientation to the data file
            thisExp.addData('Gabor.ori', Gabor.ori)
            # Add threshold, slope, and lapse rate to the data file
            thisExp.addData('QP_Threshold', threshold)
            thisExp.addData('QP_Slope', slope)
            thisExp.addData('QP_Lapse', lapse_rate)
            # Add opacity of each cue to the data file
            thisExp.addData('Left_Cue.opacity', Left_Cue.opacity)
            thisExp.addData('Right_Cue.opacity', Right_Cue.opacity)

####### NO RESPONSE TRIALS RESPONSE CHECK #################################################################################################################################################################################################### 

            # check responses
            if key_resp.keys in ['', [], None]:  # No response was made
                key_resp.keys = None
                response = None
            else:
                # Update QuestPlus with trial accuracy; QP is not updated if no response is made
                response = 1 if (
                    (key_resp.keys == '1' and thisTrial_Rep['orientation'] == 0) or 
                    (key_resp.keys == '2' and thisTrial_Rep['orientation'] == 90)
                    ) else 0 
                current_qp.update(stim={'intensity': intensity}, outcome={'response': response})
            
            # Save response key, accuracy, RT, and duration to data file
            Trial_Rep.addData('key_resp.keys',key_resp.keys)
            Trial_Rep.addData('Accuracy', response)
            if key_resp.keys != None: 
                Trial_Rep.addData('key_resp.rt', key_resp.rt)
                Trial_Rep.addData('key_resp.duration', key_resp.duration)
            
            print("Response:", response)

            # the Routine "trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()

            # Participant can take a break every 32 trials
            if (Trial_Rep.thisN+1) % 32 == 0 and Trial_Rep.thisN != 191: 
                break_text = visual.TextStim(win, text="Take a quick break!", color="black", units='pix', height=40, wrapWidth=1700)
                break_text.draw()
                win.flip()
                event.waitKeys(keyList=['space'])
                core.wait(1) 

        # Completed all experiment trials

####### PSYCHOPY ENDS EXPERIMENT AND SAVES FILE #################################################################################################################################################################################################### 

    # --- Prepare to start Routine "End" ---
    # create an object to store info about Routine End
    End = data.Routine(
        name='End',
        components=[endText, endKey],
    )
    End.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for endKey
    endKey.keys = []
    endKey.rt = []
    _endKey_allKeys = []
    # store start times for End
    End.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    End.tStart = globalClock.getTime(format='float')
    End.status = STARTED
    thisExp.addData('End.started', End.tStart)
    End.maxDuration = None
    # keep track of which components have finished
    EndComponents = End.components
    for thisComponent in End.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "End" ---
    End.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *endText* updates
        
        # if endText is starting this frame...
        if endText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            endText.frameNStart = frameN  # exact frame index
            endText.tStart = t  # local t and not account for scr refresh
            endText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(endText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'endText.started')
            # update status
            endText.status = STARTED
            endText.setAutoDraw(True)
        
        # if endText is active this frame...
        if endText.status == STARTED:
            # update params
            pass
        
        # *endKey* updates
        waitOnFlip = False
        
        # if endKey is starting this frame...
        if endKey.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            endKey.frameNStart = frameN  # exact frame index
            endKey.tStart = t  # local t and not account for scr refresh
            endKey.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(endKey, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'endKey.started')
            # update status
            endKey.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(endKey.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(endKey.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if endKey.status == STARTED and not waitOnFlip:
            theseKeys = endKey.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _endKey_allKeys.extend(theseKeys)
            if len(_endKey_allKeys):
                endKey.keys = _endKey_allKeys[-1].name  # just the last key pressed
                endKey.rt = _endKey_allKeys[-1].rt
                endKey.duration = _endKey_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            End.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in End.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "End" ---
    for thisComponent in End.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for End
    End.tStop = globalClock.getTime(format='float')
    End.tStopRefresh = tThisFlipGlobal
    thisExp.addData('End.stopped', End.tStop)
    thisExp.nextEntry()
    # the Routine "End" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)    

def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
