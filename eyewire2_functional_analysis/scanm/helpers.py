# ----------------------------------------------------------------------------
# helpers.py
#
# The MIT License (MIT)
# (c) Copyright 23 Thomas Euler, Jonathan Oesterle
#
# 2023-08-29, first implementation
# ----------------------------------------------------------------------------

def gen_scmf_dict():
    """Generate a dictionary of the ScanM file format."""
    import numpy
    return {'ComputerName': [numpy.character, 1, 'n/a'],
            'UserName': [numpy.character, 1, 'n/a'],
            'OriginalPixelDataFileName': [numpy.character, 1, ''],
            'DateStamp': [numpy.character, 1, '2023-01-01'],
            'TimeStamp': [numpy.character, 1, '00-00-1-000'],
            # 'ScanMproductVersionAndTargetOS': [numpy.character, 1, 'n/a'],
            # 'CallingProcessPath': [numpy.character, 1, ''],
            # 'CallingProcessVersion': [numpy.character, 1, 'n/a'],
            'PixelSizeInBytes': [numpy.uint32, 1, 2], 
            'StimulusChannelMask': [numpy.uint32, 1, 7],
            # 'MinVoltsAO': [numpy.float64, 1, -4.0],
            # 'MaxVoltsAO': [numpy.float64, 1, 4.0],
            # 'MaxStimulusBufferMapLength': [numpy.uint32, 1, 1],
            # 'NumberOfStimulusBuffers': [numpy.uint32, 1, 3],
            'InputChannelMask': [numpy.uint32, 1, 7],
            'TargetedPixelDuration_µs': [numpy.float64, 1, 0.],
            'RealPixelDuration_µs': [numpy.float64, 1, 0.],
            # 'Oversampling_Factor': [numpy.uint32, 1, 1],
            # 'MinVoltsAI': [numpy.float64, 1, -1.0],
            # 'MaxVoltsAI': [numpy.float64, 1, 5.0],
            'NumberOfFrames': [numpy.uint32, 1, 1],
            'ScanMode': [numpy.uint32, 1, 0],
            'FrameWidth': [numpy.uint32, 1, 0],
            'FrameHeight': [numpy.uint32, 1, 4],
            # 'ScanPathFunc': [numpy.character, 0, []],
            'PixRetraceLen': [numpy.uint32, 1, 0],
            'XPixLineOffs': [numpy.uint32, 1, 0],
            'ChunksPerFrame': [numpy.uint32, 1, 1],
            'ScanType': [numpy.uint32, 1, -1],
            # 'NSubPixOversamp': [numpy.uint32, 1, 1],
            'Zoom': [numpy.float64, 1, 1.0],
            'Angle_deg': [numpy.float64, 1, 0.0],
            # 'IgorGUIVer': [numpy.character, 1, 'n/a'],
            # 'XCoord_um': [numpy.float64, 1, 0.0],
            # 'YCoord_um': [numpy.float64, 1, 0.0],
            # 'ZCoord_um': [numpy.float64, 1, 0.0],
            # 'ZStep_um': [numpy.float64, 1, 1.0],
            # 'NFrPerStep': [numpy.uint32, 1, 1],
            # 'XOffset_V': [numpy.float64, 1, 0.0],
            # 'YOffset_V': [numpy.float64, 1, 0.0],
            # 'dZPixels': [numpy.uint32, 1, 0],
            # 'ZPixRetraceLen': [numpy.uint32, 1, 0],
            # 'ZPixLineOffs': [numpy.uint32, 1, 0],
            # 'UsesZForFastScan': [numpy.uint32, 1, 0],
            'Comment': [numpy.character, 1, 'n/a'],
            # 'SetupID': [numpy.uint32, 1, 1],
            # 'LaserWavelength_nm': [numpy.uint32, 1, 0],
            # 'Objective': [numpy.character, 1, 'n/a'],
            # 'ZLensScaler': [numpy.uint32, 1, 1],
            # 'ZLensShifty': [numpy.float64, 1, None],
            'AspectRatioFrame': [numpy.float64, 1, 1],
            # 'StimBufPerFr': [numpy.uint32, 1, 1],
            'FrameCounter': [numpy.uint32, 1, 1],
            # 'UnusedValue': [numpy.uint32, 1, 6428160],
            # 'HeaderLengthInValuePairs': [numpy.uint64, 1, 71],
            # 'Header_length_in_bytes': [numpy.uint64, 1, 5362],
            # 'StimBufLenList': [numpy.uint32, 3, array([5120, 5120, 5120])],
            # 'TargetedStimDurList': [numpy.float64, 3, array([128000., 128000., 128000.])],
            # 'RealStimDurList': [numpy.float64, 3, array([128000., 128000., 128000.])],
            # 'StimBufMapEntries': [numpy.uint64, (32, 128), []],
            'NumberOfInputChans': [numpy.uint32, 1, 1],
            # 'InChan_PixBufLenList': [numpy.uint32, 3, array([2560, 2560, 2560])],
            'nImgPerFr': [numpy.uint32, 1, 1],
            'dxFrDecoded': [numpy.uint32, 1, 80],
            'dyFrDecoded': [numpy.uint32, 1, 64],
            'dzFrDecoded': [numpy.uint32, 1, 0],
            # 'NumberOfPixBufsSet': [numpy.uint32, 1, 1500],
            # 'PixBufCounter': [numpy.uint32, 1, 388]
            }

# ----------------------------------------------------------------------------
