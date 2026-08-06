# ----------------------------------------------------------------------------
# scanm_global.py
# Global definitions for Python version of ScanM file reader
#
# The MIT License (MIT)
# (c) Copyright 2022-23 Thomas Euler, Jonathan Oesterle
#
# 2022-01-30, first implementation
# 2023-06-16, changes to cope with older files
# ----------------------------------------------------------------------------
import struct
import warnings
from enum import Enum

# pylint: disable=bad-whitespace
SCMIO_VERBOSE = 0

SCMIO_pixelDataFileExtStr = "smp"
SCMIO_headerFileExtStr = "smh"
SCMIO_configSetFileExtStr = "scmcfs"

SCMIO_typeKeySep = ","
SCMIO_keyValueSep = "="
SCMIO_keySubValueSep = ":"
SCMIO_entrySep = ";"
SCMIO_subEntrySep = "|"
SCMIO_entryFormatStr = "{0},{1}s={2};"
SCMIO_uint32Str = "UINT32"
SCMIO_uint64Str = "UINT64"
SCMIO_stringStr = "String"
SCMIO_real32Str = "REAL32"
'''
SCMIO_pixDataWaveRawFormat    = "wDataCh{0:d}_raw"
SCMIO_pixDataWaveDecodeFormat = "wDataCh{0:d}"
'''
SCMIO_maxStimBufMapEntries = 128
SCMIO_maxStimChans = 32
SCMIO_maxInputChans = 4
SCMIO_maxTrajParams = 100

ScM_scanMode_XYImage = 0
ScM_scanMode_Line = 1
ScM_scanMode_Traject = 2
ScM_scanMode_XYZImage = 3  # xy planes stacked along z
ScM_scanMode_XZYImage = 4  # xz sections stacked along y (x is fastest scanner)
ScM_scanMode_ZXYImage = 5  # zx sections stacked along y (z is fastest scanner)
ScM_scanMode_TrajectArb = 6
ScM_scanMode_XZImage = 7  # ??

ScM_scanModeStr = [
    "XYImage", "Line",
    "Traject",
    "XYZImage", "XZYImage", "ZXYImage",
    "TrajectArb",
    "XZImage"
]
# ...
ScM_scanType_timelapsed = 10
ScM_scanType_zStack = 11
# ...

ScM_simpleScanFuncList = ["XYScan2", "XYScan3", "XYZScan1", "xzSlice"]

SCMIO_preHeaderSize_bytes = 64

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Other definitions
ScM_TTLlow = 0
ScM_TTLhigh = 5

ScM_indexScannerX = 0
ScM_indexScannerY = 1
ScM_indexLaserBlk = 2
ScM_indexLensZ = 3

ScM_PixDataResorted = 0
ScM_PixDataDecoded = 1

ScM_ScanSeqParamWaveName = "wScanSeqParams"
ScM_ScanWarpParamWaveName = "wScanWarpParams"
ScM_ScanWarpParamWaveLen = 13

ScM_warpMode_None = 0
ScM_warpMode_Banana = 1
ScM_warpMode_Sector = 2
ScM_warpMode_gap = 3
ScM_warpMode_zBiCorrect = 4
ScM_warpMode_last = 5

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Error codes
ERR_Ok = 0
ERR_NotImplemented = 1
ERR_FileNotFound = 2
ERR_InvalidSMHObject = 3
ERR_SMH_NoParametersFound = 4
ERR_CannotReshapePixelData = 5
ERR_UnknownScanMode = 6
Err_SMH_NoParametersFound = 7

ERRStr = [
    "Ok",
    "`{0}` not implemented",
    "File `{0}` not found",
    "Invalid .smh object",
    ".smh parameter not found",
    "Cannot reshape pixel data",
    "Unknown scan mode"
]


# pylint: enable=bad-whitespace

# ----------------------------------------------------------------------------------
# Variable type abbreviations:
# s=string, u=uint32, f=float(real32), ...
#
# pylint: disable=bad-whitespace
class SCMIO_keys(Enum):
    ComputerName = "ComputerName"
    UserName = "UserName"
    OrigPixDataFName = "OriginalPixelDataFileName"
    DateStamp_d_m_y = "DateStamp"
    TimeStamp_h_m_s_ms = "TimeStamp"
    ScM_ProdVer_TargetOS = "ScanMproductVersionAndTargetOS"
    CallingProcessPath = "CallingProcessPath"
    CallingProcessVer = "CallingProcessVersion"
    PixelSizeInBytes = "PixelSizeInBytes"
    StimulusChannelMask = "StimulusChannelMask"
    MinVolts_AO = "MinVoltsAO"
    MaxVolts_AO = "MaxVoltsAO"
    MaxStimBufMapLen = "MaxStimulusBufferMapLength"
    NumberOfStimBufs = "NumberOfStimulusBuffers"
    InputChannelMask = "InputChannelMask"
    TargetedPixDur = "TargetedPixelDuration_µs"
    MinVolts_AI = "MinVoltsAI"
    MaxVolts_AI = "MaxVoltsAI"
    NumberOfFrames = "NumberOfFrames"
    PixelOffset = "PixelOffset"
    HdrLenInValuePairs = "HeaderLengthInValuePairs"
    HdrLenInBytes = "Header_length_in_bytes"
    FrameCounter = "FrameCounter"
    Unused0 = "UnusedValue"
    RealPixDur = "RealPixelDuration_µs"
    OversampFactor = "Oversampling_Factor"
    '''
    XCoord_um = "XCoord_um"
    YCoord_um = "YCoord_um"
    ZCoord_um = "ZCoord_um"
    ZStep_um = "ZStep_um"
    '''
    StimBufLenList = "StimBufLenList"
    StimBufMapEntries = "StimBufMapEntries"
    TargetedStimDurList = "TargetedStimDurList"
    RealStimDurList = "RealStimDurList"
    NumberOfInputChans = "NumberOfInputChans"
    InChan_PixBufLenList = "InChan_PixBufLenList"

    NumberOfPixBufsSet = "NumberOfPixBufsSet"
    PixBufCounter = "PixBufCounter"

    # --> User keys start here:
    # (SCMIO_UserParameterCount = 43)
    #
    USER_ScanMode = "ScanMode"
    USER_ScanType = "ScanType"
    USER_dxPix = "FrameWidth"
    USER_dyPix = "FrameHeight"
    USER_dzPix = "dZPixels"
    USER_scanPathFunc = "ScanPathFunc"
    USER_nPixRetrace = "PixRetraceLen"
    USER_nXPixLineOffs = "XPixLineOffs"
    USER_nYPixLineOffs = "YPixLineOffs"
    USER_nZPixLineOffs = "ZPixLineOffs"
    USER_divFrameBufReq = "ChunksPerFrame"
    USER_nSubPixOversamp = "NSubPixOversamp"
    USER_coordX = "XCoord_um"
    USER_coordY = "YCoord_um"
    USER_coordZ = "ZCoord_um"
    USER_dZStep_um = "ZStep_um"
    USER_zoom = "Zoom"
    USER_angle_deg = "Angle_deg"
    USER_IgorGUIVer = "IgorGUIVer"
    USER_NFrPerStep = "NFrPerStep"
    USER_offsetX_V = "XOffset_V"
    USER_offsetY_V = "YOffset_V"
    '''
    USER_nZPixRetrace = "ZPixRetraceLen"
    USER_usesZForFastScan = "UsesZForFastScan"
    '''
    USER_Comment = "Comment"
    USER_SetupID = "SetupID"
    USER_LaserWavelen_nm = "LaserWavelength_nm"
    USER_Objective = "Objective"
    '''
    USER_ZLensScaler = "ZLensScaler"
    USER_ZLensShifty = "ZLensShifty"
    '''
    USER_aspectRatioFrame = "AspectRatioFrame"
    USER_stimBufPerFr = "StimBufPerFr"
    USER_iChFastScan = "iChFastScan"
    '''
    USER_noYScan = "noYScan"
    '''
    USER_dxFrDecoded = "dxFrDecoded"
    USER_dyFrDecoded = "dyFrDecoded"
    USER_dzFrDecoded = "dzFrDecoded"
    USER_trajDefVRange_V = "trajDefVRange_V"
    USER_nTrajParams = "nTrajParams"
    '''
    USER_trajParams_mask = "TrajParams_*"
    '''
    USER_zoomZ = "zoomFactorZ"
    USER_offsetZ_V = "offsetZ_V"
    USER_zeroZ_V = "zeroZ_V"
    USER_ETL_polarity = "ETL_polarity_V"
    USER_ETL_minV = "ETL_min_V"
    USER_ETL_maxV = "ETL_max_V"
    USER_ETL_neutralV = "ETL_neutral_V"
    USER_nImgPerFr = "nImgPerFr"
    USER_nWarpParams = "nWarpParams"
    USER_WarpParamsStr = "WarpParamsStr"

    # For older files:
    USER_dzFr = "dZPixels"
    # <-- End of user keys


# Format strings to access the parameter sets that are stored in multiple key-value
# pairs and are here condensed into arrays
SCMIO_key_StimBufLen_x = "StimulusBufferLength_#{0}"
SCMIO_key_Ch_x_TargetedStimDur = "Channel_{0}_TargetedStimulusDuration_µs"
SCMIO_key_AO_x_Ch_x_RealStimDur = "AO_{0}_Channel_{1}_RealStimulusDuration_µs"
SCMIO_key_Ch_x_StimBufMapEntr_y = "Channel_{0}_StimulusBufferMapEntry_#{1}"
SCMIO_key_InputCh_x_PixBufLen = "PixelBuffer_#{0}_Length"
SCMIO_key_USER_trajParams_x = "TrajParams_{0}"
# pylint: enable=bad-whitespace

'''
// ----------------------------------------------------------------------------------
strconstant    ScM_usr_AutoScale        = "usr_AutoScale"

function  ScM_UserAutoScaleFunc (wFrame, scaler, newMin, newMax)
  WAVE    wFrame
  variable  scaler
  variable  &newMin, &newMax
end

// ----------------------------------------------------------------------------------
constant    SCMIO_Param_addCFDNote        = 0
constant    SCMIO_Param_integrStim        = 1
constant    SCMIO_Param_integrStim_StimCh    = 2
constant    SCMIO_Param_integrStim_TargetCh  = 3
constant    SCMIO_Param_Stim_toFractOfMax    = 4
constant    SCMIO_Param_to8Bits        = 5
constant    SCMIO_Param_cropToPixelArea    = 6
constant    SCMIO_Param_despiralTraject    = 7
constant    SCMIO_Param_lastEntry        = 7

strconstant    SCMIO_mne_ToggleIntegrStim    = " Integrate stimulus in AI3 into AI0"
'''


# ----------------------------------------------------------------------------------
def scm_load_pre_header(fPath, offset=0):
    """ Load pre-header of file `fPath` (complete path w/ extention)

        Structure   SMP_preHeaderStruct    // "uint64"
          char      fileTypeID[8]          // #0
          uint32    GUID[4]                // #1,2
          uint32    headerSize_bytes[2]    // #3
          uint32    headerLen_pairs[2]     // #4
          uint32    headerStart_bytes[2]   // #5
          uint32    pixelDataLen_bytes[2]  // #6
          uint32    analogDataLen_bytes[2] // #7
        EndStructure
    """
    d = dict()
    # Open file in binary mode to get pre-header
    with open(fPath, "rb") as f:
        if offset > 0:
            f.seek(offset)
        buf = f.read(SCMIO_preHeaderSize_bytes)
        hdr = struct.unpack("4H16s5Q", buf)
        d.update({"fileType": decode_file_type(hdr[0:3])})
        d.update({"GUID": bytearray(hdr[4]).hex()})
        d.update({"headerLen_byte": hdr[5]})
        d.update({"headerLen_values": hdr[6]})
        d.update({"headerStart_bytes": hdr[7]})
        d.update({"pixDataLen_byte": hdr[8]})
        d.update({"analogDataLen_byte": hdr[9]})
    return d


def decode_file_type(msg):
    """Decode a raw byte sequence representing a ScanM file-type identifier.

    Args:
        msg: Byte sequence (e.g. a tuple of integers) to decode as UTF-8.

    Returns:
        str: The decoded file-type string, or ``'unknown'`` if decoding fails.
    """
    try:
        return bytearray(msg).decode("utf-8")
    except Exception as e:
        warnings.warn(str(e))
        return "unknown"


def scm_log(msg, lf=True):
    """ Logs a message, currently simply by printing it to the history
    """
    if lf:
        print(msg)
    else:
        print(msg, end="")

# ----------------------------------------------------------------------------------
