# ----------------------------------------------------------------------------
# scanm_smh.py
# Class for ScanM header files (`.smh`)
#
# The MIT License (MIT)
# (c) Copyright 2022-23 Thomas Euler, Jonathan Oesterle
#
# 2022-01-30, first implementation
# 2023-06-16, changes to cope with older files
# ----------------------------------------------------------------------------
import os.path

import numpy as np

from .scanm_global import *


# ----------------------------------------------------------------------------
class SMH(object):
    """ Loads an `.smh` ScanM header file
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        """ Resets object
        """
        self._SMHPreHdrDict = {}
        self._kvPairDict = {}
        self._fPath = ""
        self._isSMHReady = False

    def loadSMH(self, fName, verbose=False):
        """ Load file `fName`
        """
        # Clear object if not empty
        if self._isSMHReady:
            self._reset()

        # Some first error checking
        errC = ERR_Ok
        fPath = os.path.splitext(fName)[0]
        fPathSMH = fPath + "." + SCMIO_headerFileExtStr
        if not os.path.exists(fPathSMH):
            errC = ERR_FileNotFound
            scm_log(ERRStr[errC].format(fPathSMH))
            return errC
        else:
            self._fPath = fPath

        scm_log(f"Processing file `{fPathSMH}`")
        try:
            # Load pre-header into a dict
            scm_log("Loading pre-header ...")
            self._SMHPreHdrDict = scm_load_pre_header(fPathSMH)

            # Load key-value pairs
            scm_log("Loading parameters (key-value pairs) ...")
            kvl = []
            key_w_err = []
            nkv = 0
            done = False

            # Read all lines from header file
            with open(fPathSMH, "rt", encoding='ISO-8859-1') as f:
                # Jump the pre-header ...
                f.seek(SCMIO_preHeaderSize_bytes)
                while not (done):
                    ln = f.readline()
                    done = len(ln) == 0
                    if not done:
                        # Line with content
                        # -> convert to normal string
                        # (this is a bit complicated because of special characters,
                        #  such as `µ`)
                        buf = ln.encode("utf-8")
                        tmp = ""
                        ich = -1
                        for j in range((len(buf) - 1) // 2):
                            if buf[ich + 1] == 0:
                                ich += 2
                                tmp += chr(buf[ich])
                            else:
                                ich += 3
                                tmp = tmp[:-1] + chr(buf[ich - 2]) + chr(buf[ich])

                        s = "".join(tmp).split("\n")[0]
                        if len(s) > 0 and s[0] != "\x00":
                            # String is not empty and nd of file not reached
                            # -> store string in list
                            if verbose:
                                scm_log(f"-> {nkv:5} {s}")
                            kvl.append(s)
                            nkv += 1
            if nkv == 0:
                errC = Err_SMH_NoParametersFound
                scm_log(ERRStr(errC))
                return errC

            # Now parse the found key-value pairs
            scm_log(f"{nkv} key-value pair(s) found")
            for i, s in enumerate(kvl):
                # Get the different elements of the key-value pair
                tmp = s.split(SCMIO_typeKeySep)
                sty = tmp[0]
                tmp = tmp[1].split(SCMIO_keyValueSep)
                svr = tmp[0].strip()
                tmp = [s.strip() for s in tmp[1].split(SCMIO_entrySep) if len(s) > 0]
                tid = None
                nvl = 1
                v = tmp[0]

                # Extract value(s) depending on type
                if sty == SCMIO_stringStr:
                    svl = v.split(SCMIO_subEntrySep)
                    if len(svl) == 1:
                        svl = svl[0]
                    else:
                        nvl = len(svl)
                    tid = np.character
                elif sty == SCMIO_real32Str:
                    svl = float(v)
                    tid = np.float64
                elif sty in [SCMIO_uint32Str, SCMIO_uint64Str]:
                    if not v.lower() == "nan":
                        try:
                            svl = int(v)
                        except ValueError:
                            # Mark key as erronous
                            svl = None
                            scm_log(f"ERROR reading `{svr}` (value is `{v}`)")
                            key_w_err.append(svr)
                        tid = np.uint32 if sty == SCMIO_uint32Str else np.uint64
                    else:
                        svl = None
                        tid = np.float64
                else:
                    raise NotImplementedError(f"Type `{sty}` not implemented")

                # Add parsed parameter to the parameter dictionary
                self._kvPairDict.update({svr: [tid, nvl, svl]})
                if verbose:
                    scm_log(f"-> {i:5} {s}")

            # Report and correct errors, if any
            if len(key_w_err) > 0:
                scm_log(f"Errors in key(s): {', '.join(key_w_err)}")
                scm_log(f"Trying to repair ... ")
                for sKey in key_w_err:
                    if sKey == SCMIO_keys.USER_nPixRetrace.value:
                        v = int(self.get(SCMIO_keys.USER_scanPathFunc)[4])
                        self._kvPairDict.update(
                            {SCMIO_keys.USER_nPixRetrace.value: [np.uint32, 1, v]
                             })
                    elif sKey == SCMIO_keys.USER_nXPixLineOffs.value:
                        v = int(self.get(SCMIO_keys.USER_scanPathFunc)[5])
                        self._kvPairDict.update(
                            {SCMIO_keys.USER_nXPixLineOffs.value: [np.uint32, 1, v]
                             })

            # Some parameters need to be processed
            stimBufLenList = []
            tarStimDurList = []
            realStimDurList = []

            nStimBuf = self.get(SCMIO_keys.NumberOfStimBufs)
            for iBuf in range(nStimBuf):
                sKey = SCMIO_key_StimBufLen_x.format(iBuf)
                v = self.get(sKey, remove=True)
                stimBufLenList.append(v)

                sKey = SCMIO_key_Ch_x_TargetedStimDur.format(iBuf)
                v = self.get(sKey, remove=True)
                tarStimDurList.append(v)

                sKey = SCMIO_key_AO_x_Ch_x_RealStimDur.format("A", iBuf)
                v = self.get(sKey, remove=True)
                realStimDurList.append(v)

            # Add the 3 lists to dict
            self._kvPairDict.update(
                {SCMIO_keys.StimBufLenList.value:
                     [np.uint32, len(stimBufLenList), np.array(stimBufLenList)]
                 })
            self._kvPairDict.update(
                {SCMIO_keys.TargetedStimDurList.value:
                     [np.float64, len(tarStimDurList), np.array(tarStimDurList)]
                 })
            self._kvPairDict.update(
                {SCMIO_keys.RealStimDurList.value:
                     [np.float64, len(tarStimDurList), np.array(realStimDurList)]
                 })

            # Retrieve stimulus buffer map
            StimBufMapEntr = np.array([[0] * SCMIO_maxStimBufMapEntries] * SCMIO_maxStimChans)
            mask = self.get(SCMIO_keys.StimulusChannelMask)
            for iCh in range(SCMIO_maxStimChans):
                if mask & (1 << iCh):
                    for iEntr in range(self.get(SCMIO_keys.MaxStimBufMapLen)):
                        sKey = SCMIO_key_Ch_x_StimBufMapEntr_y.format(iCh, iEntr)
                        v = self.get(sKey, remove=True)
                        StimBufMapEntr[iCh][iEntr] = v
            self._kvPairDict.update(
                {SCMIO_keys.StimBufMapEntries.value:
                     [np.uint64, np.shape(StimBufMapEntr), StimBufMapEntr]
                 })

            # Retrieve list of input channel pixel buffer lengths
            # (number of pixel puffer is continous, NOT equal to the AI channel index!)
            nInCh = 0
            pixBufLenList = []
            mask = self.get(SCMIO_keys.InputChannelMask)
            for iInCh in range(SCMIO_maxInputChans):
                if mask & (1 << iInCh):
                    sKey = SCMIO_key_InputCh_x_PixBufLen.format(nInCh)
                    v = self.get(sKey, remove=True)
                    pixBufLenList.append(v)
                    nInCh += 1
            self._kvPairDict.update(
                {SCMIO_keys.NumberOfInputChans.value:
                     [np.uint32, 1, nInCh]
                 })
            self._kvPairDict.update(
                {SCMIO_keys.InChan_PixBufLenList.value:
                     [np.uint32, len(pixBufLenList), np.array(pixBufLenList)]
                 })

            # Add some keys if they are not yet defined
            if self.nImgPerFr is None:
                self._kvPairDict.update(
                    {SCMIO_keys.USER_nImgPerFr.value: [np.uint32, 1, 1]
                     })
            if self.dxFrDec_pix is None:
                self._kvPairDict.update(
                    {SCMIO_keys.USER_dxFrDecoded.value: [np.uint32, 1, 0]
                     })
            if self.dyFrDec_pix is None:
                self._kvPairDict.update(
                    {SCMIO_keys.USER_dyFrDecoded.value: [np.uint32, 1, 0]
                     })
            if self.dzFrDec_pix is None:
                self._kvPairDict.update(
                    {SCMIO_keys.USER_dzFrDecoded.value: [np.uint32, 1, 0]
                     })
            if self.dzFr_pix is None:
                self._kvPairDict.update(
                    {SCMIO_keys.USER_dzFr.value: [np.uint32, 1, 1]
                     })
            if self.nPixBufsSet is None:
                v = self.get(SCMIO_keys.NumberOfFrames)
                self._kvPairDict.update(
                    {SCMIO_keys.NumberOfPixBufsSet.value: [np.uint32, 1, v]
                     })
            if self.pixBufCounter is None:
                v = self.get(SCMIO_keys.FrameCounter)
                self._kvPairDict.update(
                    {SCMIO_keys.PixBufCounter.value: [np.uint32, 1, v]
                     })
            if self.nStimBufPerFr is None:
                self._kvPairDict.update(
                    {SCMIO_keys.USER_stimBufPerFr.value: [np.uint32, 1, 1]
                     })
            if self.aspectRatioFr is None:
                self._kvPairDict.update(
                    {SCMIO_keys.USER_aspectRatioFrame.value: [np.uint32, 1, 1]
                     })

                # Stay compatible with older data files
            scm_log(f"Correct parameters for older files ...")
            if self.scanMode == ScM_scanMode_XYImage:
                if (self.dxFr_pix / (self.dxRetrace_pix + self.dxOffs_pix)) < 4:
                    self.set(
                        SCMIO_keys.USER_nPixRetrace,
                        self.get(SCMIO_keys.USER_scanPathFunc)[3 + 1]
                    )
                    self.set(
                        SCMIO_keys.USER_nXPixLineOffs,
                        self.get(SCMIO_keys.USER_scanPathFunc)[4 + 1]
                    )
                if self.dxFrDec_pix == 0:
                    self.set(SCMIO_keys.USER_dxFrDecoded, self.dxFr_pix)
                if self.dyFrDec_pix == 0:
                    self.set(SCMIO_keys.USER_dyFrDecoded, self.dyFr_pix)
                if self.dzFrDec_pix == 0:
                    self.set(SCMIO_keys.USER_dzFrDecoded, self.dzFr_pix)

            self.set(
                SCMIO_keys.USER_aspectRatioFrame,
                max(1, self.get(SCMIO_keys.USER_aspectRatioFrame))
            )
            self.set(
                SCMIO_keys.USER_stimBufPerFr,
                max(1, self.get(SCMIO_keys.USER_stimBufPerFr))
            )
            self.set(
                SCMIO_keys.USER_nImgPerFr,
                max(1, self.nImgPerFr)
            )

            scm_log(f"{len(self._kvPairDict)} parameter(s) extracted")
            self._isSMHReady = True
            scm_log("Done.")

        except:
            raise
        return errC

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ''' General information
    '''

    @property
    def GUID(self):
        """str: GUID hex string read from the SMH pre-header."""
        return self._SMHPreHdrDict["GUID"]

    @property
    def filePath(self):
        """str: Base file path (without extension) of the loaded SMH file."""
        return self._fPath

    @property
    def isSMHReady(self):
        """bool: ``True`` if an SMH file has been successfully loaded."""
        return self._isSMHReady

    ''' Scan mode and type
    '''

    @property
    def scanMode(self):
        """int: ScanM scan mode code (e.g. ``ScM_scanMode_XYImage = 0``)."""
        return self.get(SCMIO_keys.USER_ScanMode)

    @property
    def scanType(self):
        """int: ScanM scan type code (e.g. ``ScM_scanType_timelapsed = 10``)."""
        return self.get(SCMIO_keys.USER_ScanType)

    ''' Pixel-related information
    '''

    @property
    def pixSize_byte(self):
        """int: Pixel data size in bytes (typically 2 or 8)."""
        return self.get(SCMIO_keys.PixelSizeInBytes)

    @property
    def pixDurTarget_us(self):
        """float: Targeted pixel dwell time in microseconds."""
        return self.get(SCMIO_keys.TargetedPixDur)

    @property
    def pixDur_us(self):
        """float: Actual (realised) pixel dwell time in microseconds."""
        return self.get(SCMIO_keys.RealPixDur)

    ''' Number of recorded frames
    '''

    @property
    def nFr(self):
        """int: Number of recorded frames, or ``-1`` if not yet determined."""
        try:
            return self._nFr
        except AttributeError:
            return -1

    ''' Zoom factor
    '''

    @property
    def zoom(self):
        """float: Optical zoom factor used during the scan."""
        return self.get(SCMIO_keys.USER_zoom)

    ''' Parameters on frame structure
    '''

    @property
    def nPixBufPerFr(self):
        """int: Number of pixel buffers per frame (``ChunksPerFrame``)."""
        return self.get(SCMIO_keys.USER_divFrameBufReq)

    @property
    def nStimBufPerFr(self):
        """int: Number of stimulus buffers per frame."""
        return self.get(SCMIO_keys.USER_stimBufPerFr)

    @property
    def nImgPerFr(self):
        """int: Number of images per frame (usually 1)."""
        return self.get(SCMIO_keys.USER_nImgPerFr)

    ''' Frame dimensions
    '''

    @property
    def dxFr_pix(self):
        """int: Frame width in pixels (fast axis)."""
        return self.get(SCMIO_keys.USER_dxPix)

    @property
    def dyFr_pix(self):
        """int: Frame height in pixels (slow axis 1)."""
        return self.get(SCMIO_keys.USER_dyPix)

    @property
    def dzFr_pix(self):
        """int: Frame depth in pixels (slow axis 2 / z)."""
        return self.get(SCMIO_keys.USER_dzPix)

    @property
    def dxOffs_pix(self):
        """int: Pixel offset along the fast (x) axis."""
        return self.get(SCMIO_keys.USER_nXPixLineOffs)

    @property
    def dxRetrace_pix(self):
        """int: Number of retrace pixels along the fast (x) axis."""
        return self.get(SCMIO_keys.USER_nPixRetrace)

    @property
    def dxFrDec_pix(self):
        """int: Decoded frame width in pixels."""
        return self.get(SCMIO_keys.USER_dxFrDecoded)

    @property
    def dyFrDec_pix(self):
        """int: Decoded frame height in pixels."""
        return self.get(SCMIO_keys.USER_dyFrDecoded)

    @dyFrDec_pix.setter
    def dyFrDec_pix(self, v):
        """Set the decoded frame height in pixels."""
        self.set(SCMIO_keys.USER_dyFrDecoded, v)

    @property
    def dzFrDec_pix(self):
        """int: Decoded frame depth in pixels."""
        return self.get(SCMIO_keys.USER_dzFrDecoded)

    @property
    def aspectRatioFr(self):
        """float: Aspect ratio of the scan frame (width / height)."""
        return self.get(SCMIO_keys.USER_aspectRatioFrame)

    ''' Stimulus-related
    '''

    @property
    def nStimBuf(self):
        """int: Number of stimulus buffers."""
        return self.get(SCMIO_keys.NumberOfStimBufs)

    @property
    def stimChMask(self):
        """int: Bitmask of active stimulus channels."""
        return self.get(SCMIO_keys.StimulusChannelMask)

    ''' Input-related
    '''

    @property
    def nInputCh(self):
        """int: Number of active analog-input channels."""
        return self.get(SCMIO_keys.NumberOfInputChans)

    @property
    def inputChMask(self):
        """int: Bitmask of active analog-input channels."""
        return self.get(SCMIO_keys.InputChannelMask)

    @property
    def nPixBufsSet(self):
        """int: Total number of pixel buffers set for this recording."""
        return self.get(SCMIO_keys.NumberOfPixBufsSet)

    @nPixBufsSet.setter
    def nPixBufsSet(self, v):
        """Set the total number of pixel buffers."""
        self.set(SCMIO_keys.NumberOfPixBufsSet, v)

    @property
    def pixBufCounter(self):
        """int: Number of pixel buffers actually recorded."""
        return self.get(SCMIO_keys.PixBufCounter)

    @pixBufCounter.setter
    def pixBufCounter(self, v):
        """Set the pixel buffer counter value."""
        self.set(SCMIO_keys.PixBufCounter, v)

    @property
    def pixBufLenList(self):
        """numpy.ndarray: Array of pixel buffer lengths per input channel."""
        return self.get(SCMIO_keys.InChan_PixBufLenList)

    '''
    'StimBufLenList': [numpy.uint32, 3, array([5120, 5120, 5120])],
    'TargetedStimDurList': [numpy.float64, 3, array([128000., 128000., 128000.])],
    'RealStimDurList': [numpy.float64, 3, array([128000., 128000., 128000.])],
    'StimBufMapEntries': [numpy.uint64, (32, 128), array([[0, 0, 0, ..., 0, 0, 0], ... [0, 0, 0, ..., 0, 0, 0]])],
      'MinVoltsAO': [numpy.float64, 1, -4.0],
    'MaxVoltsAO': [numpy.float64, 1, 4.0],
    'MaxStimulusBufferMapLength': [numpy.uint32, 1, 1],
    'Oversampling_Factor': [numpy.uint32, 1, 25],
    'MinVoltsAI': [numpy.float64, 1, -1.0],
    'MaxVoltsAI': [numpy.float64, 1, 5.0],
    'ScanPathFunc': [numpy.character,   8,   ['XYScan2', '5120', '80', '64', '10', '6', '0', '1']],
    'NSubPixOversamp': [numpy.uint32, 1, 25],
    'Angle_deg': [numpy.float64, 1, 0.0],
    'IgorGUIVer': [numpy.character, 1, '0.0.38.02'],
    'XCoord_um': [numpy.float64, 1, -599.52],
    'YCoord_um': [numpy.float64, 1, 881.16],
    'ZCoord_um': [numpy.float64, 1, 6213.9],
    'ZStep_um': [numpy.float64, 1, 1.0],
    'NFrPerStep': [numpy.uint32, 1, 1],
    'XOffset_V': [numpy.float64, 1, 0.0],
    'YOffset_V': [numpy.float64, 1, 0.0],
    'ZPixRetraceLen': [numpy.uint32, 1, 0],
    'ZPixLineOffs': [numpy.uint32, 1, 0],
    'UsesZForFastScan': [numpy.uint32, 1, 0],
    'Comment': [numpy.character, 1, 'n/a'],
    'SetupID': [numpy.uint32, 1, 1],
    'LaserWavelength_nm': [numpy.uint32, 1, 0],
    'Objective': [numpy.character, 1, 'n/a'],
    'ZLensScaler': [numpy.uint32, 1, 1],
    'ZLensShifty': [None, 1, 1],
    'UnusedValue': [numpy.uint32, 1, 6428160],
    'HeaderLengthInValuePairs': [numpy.uint64, 1, 71],
    'Header_length_in_bytes': [numpy.uint64, 1, 5362],
    '''

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def summary(self):
        """Print a human-readable summary of the key SMH header parameters to stdout."""
        print(f"Summary")
        print(f"-------")
        s = ScM_scanModeStr[self.scanMode]
        print(f"Scan    : mode, type   : {s} ({self.scanMode}), {self.scanType}")
        print(f"Pixel   : size         : {self.pixSize_byte} bytes/pixel")
        print(f"          duration     : {self.pixDur_us} us ({self.pixDurTarget_us})")
        print(f"Frame   : x-y size     : {self.dxFr_pix} x {self.dyFr_pix} pixels")
        print(f"          x-offset     : {self.dxOffs_pix} pixels")
        print(f"          x-retrace    : {self.dxRetrace_pix} pixels")
        s = f"{self.nFr} recorded" if self.nFr >= 0 else "n/a"
        print(f"          count        : {s}")
        print(f"          organisation : {self.nPixBufPerFr} pixel buffers/frame")
        print(f"Stimulus: # of buffers : {self.nStimBuf}")
        print(f"          mask         : {self.stimChMask:04b}")
        print(f"Input   : # of channels: {self.nInputCh}")
        print(f"          mask         : {self.inputChMask:04b}")
        print(f"Zoom factor            : {self.zoom:3.2f}")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def get(self, key, index=-1, remove=False):
        """ Get value for given key
        """
        key = key if isinstance(key, str) else key.value
        try:
            val = self._kvPairDict[key]
            if index < 0:
                res = val[2]
            else:
                if isinstance(val[1], tuple):
                    assert index >= 0 and index < val[1][0], "Index out of range"
                    res = val[2][index]
                else:
                    assert index >= 0 and index < val[1], "Index out of range"
                    res = val[2] if val[1] == 1 else val[2][index]
            if remove:
                _ = self._kvPairDict.pop(key)
            return res
        except KeyError:
            return None

    def set(self, key, val):
        """ Set new value for given key
        """
        key = key if isinstance(key, str) else key.value
        try:
            data = self._kvPairDict[key]
            data[2] = val
            self._kvPairDict[key] = data
        except KeyError:
            scm_log(f"ERROR: Key `{key}` not found ")

# ----------------------------------------------------------------------------
