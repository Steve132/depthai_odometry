#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import json
from pprint import pprint

# https: // docs.luxonis.com/projects/api/en/latest/samples/calibration/calibration_reader/


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

    useImu = False
    if(useImu):
        imu = pipeline.create(dai.node.IMU)
        imu_xout = pipeline.create(dai.node.XLinkOut)
        imu_xout.setStreamName("imu")
        imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 120)
        imu.enableIMUSensor(dai.IMUSensor.LINEAR_ACCELERATION, 120)
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)
        imu.out.link(imu_xout.input)

    # Define sources and outputswell
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
    stereo.initialConfig.setDepthAlign(
        dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)
    stereo.setRectification(True)
    stereo.setRectifyEdgeFillColor(0)
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
        if(useImu):
            qimu = device.getOutputQueue(name="imu", maxSize=2, blocking=False)

        calibData = device.readCalibration()
        intrinsics = calibData.getCameraIntrinsics(
            dai.CameraBoardSocket.LEFT, 640, 400)
        pprint(intrinsics)

        while True:
            leftEvent = qleft.get()
            depthEvent = qdepth.get()

            if(useImu):
                imuEvent = qimu.get()
                imuPacket = imuEvent.packets[-1]
                rot = imuPacket.rotationVector
                acc = imuPacket.imuPacket.acceleroMeter
                print(rot)
                print(acc)

            leftImg = leftEvent.getCvFrame()
            depthImg = depthEvent.getCvFrame()
            depthImg = depthImg.astype(np.float32)/1000.0

            yield leftImg, depthImg, intrinsics


def computeMask(depthImg):
    return np.array(depthImg != 0, dtype=np.uint8)


def dispLD(leftImg, depthImg, mask):
    # mx, depthImg = cv2.threshold(depthImg, thresh=6000,
    #                             maxval=16000, type=cv2.THRESH_TRUNC)
    # depthImg *= ((1 << 16) // 6000)
    cv2.imshow("left", leftImg)
    cv2.imshow("depth", depthImg)
    cv2.imshow("mask", mask)
    return cv2.waitKey(1) != ord('q')


def main():
    cur_transform = np.eye(4)
    prev_frame = None
    # pprint(dir(cv2.rgbd))
#    odo = cv2.rgbd_RgbdICPOdometry()
    odo = cv2.rgbd.RgbdICPOdometry_create()
    odo.setTransformType(cv2.rgbd.ODOMETRY_RIGID_BODY_MOTION)
    odo.setMaxDepth(32.0)
    odo.setMinDepth(0.001)
    # return
    rgbd_stream = getRgbd()

    for leftImg, depthImg, intrin in getRgbd():
        mask = computeMask(depthImg)
        if(prev_frame == None):
            prev_frame = (leftImg, depthImg, mask)
            odo.setCameraMatrix(np.array(intrin))
            continue
        dispLD(leftImg, depthImg, mask)
        found, tform = odo.compute(leftImg, depthImg, mask,
                                   prev_frame[0], prev_frame[1], prev_frame[2])

        if(found):
            cur_transform = cur_transform @ tform
            print(json.dumps(list([list(x) for x in cur_transform])))
            prev_frame = (leftImg, depthImg, mask)


main()
