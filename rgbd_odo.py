#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np


def getRgbd():
    # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
    extended_disparity = False
    # Better accuracy for longer distance, fractional disparity 32-levels:
    subpixel = True
    # Better handling for occlusions:
    lr_check = True
    fps = None
    confidenceThreshold = None  # 200
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo_xout = pipeline.create(dai.node.XLinkOut)
    left_xout = pipeline.create(dai.node.XLinkOut)

    stereo_xout.setStreamName("depth")
    left_xout.setStreamName("left")

    # Properties
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    if(fps):
        monoLeft.setFps(fps)
    monoRight.setResolution(
        dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    if(fps):
        monoRight.setFps(fps)

    # Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
    stereo.setDefaultProfilePreset(
        dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    if(confidenceThreshold):
        stereo.initialConfig.setConfidenceThreshold(confidenceThreshold)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
    # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
    stereo.setLeftRightCheck(lr_check)
    stereo.setExtendedDisparity(extended_disparity)
    stereo.setSubpixel(subpixel)

    config = stereo.initialConfig.get()
    config.postProcessing.speckleFilter.enable = False
    config.postProcessing.speckleFilter.speckleRange = 50
    config.postProcessing.temporalFilter.enable = False
    config.postProcessing.spatialFilter.enable = False
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 1
    config.postProcessing.thresholdFilter.minRange = 200
    config.postProcessing.thresholdFilter.maxRange = 13000
    config.postProcessing.decimationFilter.decimationFactor = 1
    stereo.initialConfig.set(config)

    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    stereo.depth.link(stereo_xout.input)
    stereo.rectifiedLeft.link(left_xout.input)
    # void setXLinkChunkSize(int sizeBytes) TODO this has to do with chunking and latency.
    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        qleft = device.getOutputQueue(name="left", maxSize=2, blocking=False)
        qdepth = device.getOutputQueue(name="depth", maxSize=2, blocking=False)
        while True:
            leftEvent = qleft.get()
            depthEvent = qdepth.get()
            leftImg = leftEvent.getCvFrame()
            depthImg = depthEvent.getCvFrame()
            mx, depthImg = cv2.threshold(depthImg, thresh=6000,
                                         maxval=16000, type=cv2.THRESH_TRUNC)
            depthImg *= ((1 << 16) // 6000)
            yield leftImg, depthImg


def dispLD(leftImg, depthImg):
    cv2.imshow("left", leftImg)
    cv2.imshow("depth", depthImg)
    return cv2.waitKey(1) != ord('q')


for leftImg, depthImg in getRgbd():
    if(not dispLD(leftImg, depthImg)):
        break
