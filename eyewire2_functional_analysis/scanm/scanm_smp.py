# -------------------------------------------------------------------------------------------
# scanm_smp.py
# Class for ScanM pixel data files (`.smp`)
#
# The MIT License (MIT)
# (c) Copyright 2022-23 Thomas Euler, Jonathan Oesterle
#
# 2022-01-31, first implementation
# 2023-06-16, changes to cope with older files
# -------------------------------------------------------------------------------------------
import os.path

import numpy as np

from .scanm_global import *
from .scanm_smh import SMH
from .scanm_stim_buf import StimBuf


# -------------------------------------------------------------------------------------------
class SMP(SMH):
    """Loads `.smp` ScanM pixel data file (for a specific `.smh` header file)
    """

    def __init__(self):
        super().__init__()
        self._reset()

    def _reset(self):
        """ Resets object
    """
        self._isSMPReady = False
        self._SMPPreHdrDict = dict()
        super()._reset()

    '''
    Inherited:
    def loadSMH(self, fName, verbose=False):  
    '''

    def loadSMP(self, verbose=False):
        """ Load pixel data file for the respective `smh` object
        """
        # Clear object if not empty
        if self._isSMPReady:
            self._reset()

        # Some first error checking
        if not self._isSMHReady:
            scm_log(f"ERROR: Load `.smh` file first")
            return ERR_InvalidSMHObject
        fPathSMP = self._fPath + "." + SCMIO_pixelDataFileExtStr
        if not os.path.exists(fPathSMP):
            scm_log(f"ERROR: File `{fPathSMP}` not found")
            return ERR_FileNotFound

        # Check for currently implemented scanModes
        if not (self.scanMode in [ScM_scanMode_XYImage, ScM_scanMode_XZYImage]):
            s = ScM_scanModeStr[self.scanMode]
            scm_log("ERROR: " + ERRStr[ERR_NotImplemented].format(s))
            return ERR_NotImplemented
        # ***************
        # ***************
        # TODO: Add code for other scan modes
        '''
        ScM_scanMode_Line             = 1
        ScM_scanMode_Traject          = 2
        ScM_scanMode_XYZImage         = 3    # xy planes stacked along z
        ScM_scanMode_XZYImage         = 4    # xz sections stacked along y (x is fastest scanner)
        ScM_scanMode_ZXYImage         = 5    # zx sections stacked along y (z is fastest scanner)
        ScM_scanMode_TrajectArb       = 6
        ScM_scanMode_XZImage          = 7    # ??
        '''
        # ***************
        # ***************

        # Get stim buffer information
        self._StimBuf = StimBuf(self)

        # Get some scanMode-related parameters
        nFrPerStep = self.get(SCMIO_keys.USER_NFrPerStep)
        isAvZStack = self.scanType == ScM_scanType_zStack and nFrPerStep > 1
        nFrPerStep = nFrPerStep if isAvZStack else 1

        errC = ERR_Ok
        scm_log(f"Processing file `{fPathSMP}`")
        try:
            # Load post-header into a dict
            # (In the pixel data file, the recorded data is followed by a 64-byte "header", which
            #  has the same structure as the pre-header at the beginning of the `.smh` file. Only
            #  if the GUIDs there match between the two files, on can be sure that the files belong
            #  together.)
            scm_log("Loading post-header ...")
            self._SMPPreHdrDict = scm_load_pre_header(
                fPathSMP, self._SMHPreHdrDict["analogDataLen_byte"]
            )
            gp = self._SMPPreHdrDict["GUID"]
            gh = self._SMHPreHdrDict["GUID"]

            if gp != gh:
                # GUIDs in header and pixel data files do not match!!!
                scm_log(f"WARNING: GUID mismatch {gh} != {gp}")

            # Prepare reading pixel data
            errC = ERR_Ok
            if self.scanMode in [ScM_scanMode_XYImage, ScM_scanMode_TrajectArb]:
                dFast = self.dxFr_pix
                nFastPixRetr = self.dxRetrace_pix
                nFastPixOff = self.dxOffs_pix
                dSlow1 = self.dyFr_pix if self.dyFr_pix > 0 else 1
                dSlow2 = self.dzFr_pix if self.dzFr_pix > 0 else 1
                if self.scanMode == ScM_scanMode_TrajectArb:
                    # TODO
                    pass

            elif self.scanMode == ScM_scanMode_XZYImage:
                dFast = self.dxFr_pix
                nFastPixRetr = self.dxRetrace_pix
                nFastPixOff = self.dxOffs_pix
                dSlow1 = self.dzFr_pix if self.dzFr_pix > 0 else 1
                dSlow2 = self.dyFr_pix if self.dyFr_pix > 0 else 1

            elif self.scanMode == ScM_scanMode_ZXYImage:
                errC = ERR_NotImplemented
                """
        dFast = pwNP[%User_dzPix]
        nFastPixRetr = pwNP[%User_nPixRetrace]
        nFastPixOff = pwNP[%User_nZPixLineOffs]
        dSlow1 = pwNP[%User_dxPix]
        dSlow2 = pwNP[%User_dyPix]
        """
            # ***************
            # ***************
            else:
                errC = ERR_UnknownScanMode

            if errC != ERR_Ok:
                s = "ERROR: " + ERRStr[errC]
                if errC == ERR_NotImplemented:
                    s = s.format(ScM_scanModeStr[self.scanMode])
                scm_log(s)
                return errC

            # Check pixel size
            assert self.pixSize_byte in [2, 8], "ABORT: Invalid pixel size"
            _dtype = np.double if self.pixSize_byte == 8 else np.uint16

            # Correct number of pixel buffers, because it is not correctly reported
            # by the ScanM.dll if one stimulus buffer contained the data for multiple
            # frames (i.e. cp.stimBufPerFr != 1)
            if self.nStimBufPerFr > 0:
                self.nPixBufsSet *= self.nStimBufPerFr
                self.pixBufCounter *= self.nStimBufPerFr

            # Determine some parameters
            pixBLen = self.pixBufLenList[0]
            nPixPerFr = dFast * dSlow1 * dSlow2
            nBufPerFr = nPixPerFr / pixBLen
            if self.nPixBufsSet == self.pixBufCounter:
                nPixB = self.nPixBufsSet * nBufPerFr
            else:
                nPixB = (self.nPixBufsSet - self.pixBufCounter) * nBufPerFr
            nPixB = int(nPixB * nFrPerStep)
            nAICh = int(self.nInputCh)
            nImgPerFr = max(1, self.nImgPerFr)
            self._nFr = int((nPixB / nFrPerStep * pixBLen) / nPixPerFr * nImgPerFr)
            assert nImgPerFr == 1, "ABORT: `nImgPerFr` larger than 1??"

            # Correct decoded frame size in case of bidrectional scans
            if nImgPerFr > 1:
                if self.scanMode == ScM_scanMode_XZYImage:
                    self.dyFrDec_pix /= nImgPerFr

            dxFrDec = self.dxFrDec_pix
            dyFrDec = self.dyFrDec_pix
            dzFrDec = self.dzFrDec_pix
            nPixDecFr = dxFrDec * dyFrDec  # *dzFrDec

            scm_log(f"{nAICh} AI channel(s) ({self.inputChMask:#04b})")
            scm_log(f"{nPixB:.0f} of {self.nPixBufsSet} buffer(s) (each {pixBLen} pixels) "
                    "per channel")

            # Make arrays for pixel data, one per AI channel
            self._hasDecoded = (
                    self._StimBuf.isExtScanFunction and
                    self._StimBuf.pixDecodeMode == ScM_PixDataDecoded
            )
            """if self._hasDecoded:
                # ***************
                # ***************
                # TODO
                # ***************
                # ***************
                scm_log("ERROR: " + ERRStr[ERR_NotImplemented].format("Pixel decoding"))
                return ERR_NotImplemented
"""
            self._wPixData = []
            self._wDataCh = []
            self._wDecode = np.zeros(nPixDecFr) if self._hasDecoded else None
            self._wDecodeAv = np.zeros(nPixDecFr) if self._hasDecoded else None

            for iInCh in range(SCMIO_maxInputChans):
                if self.inputChMask & (2 ** iInCh):
                    # Original explanation:
                    # "wDataChx_raw" contains the raw pixel data, "wDataChx" the decoded pixel
                    # data that can be directly used for traditional preprocessing. With standard
                    # xy scans, raw and decoded are the same (or for scan path functions that
                    # just resort the pixel data), therefore no "wDataChx_raw" wave
                    # are created
                    # -> pwPixData
                    n = int(nPixB / nFrPerStep * pixBLen)
                    self._wPixData.append([iInCh, np.zeros(n, _dtype)])
                    if self._hasDecoded:
                        self._wDataCh.append([iInCh, np.zeros((dxFrDec, dyFrDec, self._nFr))])



            # Read pixel data
            with open(fPathSMP, "rb") as f:
                # Load pixel data buffer by buffer in the AI channel waves
                iPixB = 0
                iPixBAllCh = 0
                iAvFr = 0
                isExtSPF = self._StimBuf.isExtScanFunction
                isDecoded = self._StimBuf.pixDecodeMode == ScM_PixDataDecoded
                wTempMoreParam = [0] * 7

                if isAvZStack:
                    # Is z-stack with more than one frame per step, requires
                    # averaging ...
                    # ***************
                    # ***************
                    # TODO
                    # ***************
                    # ***************
                    scm_log("ERROR: " + ERRStr[ERR_NotImplemented].format("Z-stacks"))
                    return ERR_NotImplemented

                else:
                    # w/o frame averaging (as usual)
                    npx = int(pixBLen * nAICh)
                    sup = f"{npx}H" if self.pixSize_byte == 2 else f"{npx}d"

                    iPixBPerCh = 0
                    eof = False
                    for iPixBPerCh in range(nPixB):
                        # Read next pixel buffer (containing all AI channels)
                        _size_req = npx * self.pixSize_byte
                        buf = f.read(_size_req)
                        _size_read = len(buf)
                        eof = _size_read < _size_req
                        if eof:
                            # End of file reached ...
                            assert False, "ABORT: End of .smp file, should not happen ..."

                        self._wPixBAllCh = np.array(struct.unpack(sup, buf))
                        iCh = 0
                        for iInCh in range(SCMIO_maxInputChans):
                            if self.inputChMask & (2 ** iInCh):
                                self._wPixB = np.array(self._wPixBAllCh[iCh * pixBLen:(iCh + 1) * pixBLen])
                                if self._hasDecoded or not (self._StimBuf.isExtScanFunction):
                                    # Standard scan path function was used or external scan path function
                                    # with pixel decoding
                                    j = 0
                                    while not self._wPixData[j][0] == iInCh and j < SCMIO_maxInputChans: j += 1
                                    m = iPixBPerCh * pixBLen
                                    n = (iPixBPerCh + 1) * pixBLen
                                    self._wPixData[j][1][m:n] = self._wPixB[:]

                                if self._StimBuf.isExtScanFunction:
                                    # External scan path function, therefore call the decoder to fill
                                    # the second set of pixel data waves
                                    # ***************
                                    # ***************
                                    # TODO
                                    # ***************
                                    # ***************
                                    scm_log("ERROR: " + ERRStr[ERR_NotImplemented].format("External decoder functions"))
                                    return ERR_NotImplemented
                                    '''
                  wTempMoreParam[0] = nAICh
                  wTempMoreParam[1] = iCh
                  wTempMoreParam[2] = iPixBAllCh *PixBLen
                  wTempMoreParam[3] = nPixPerFr	 /nImgPerFr
                  wTempMoreParam[4] = PixBLen
                  wTempMoreParam[5] = 1
                  wTempMoreParam[6] = 0
                  fDecode(wDecode, wDecodeAv, pwPixBAllCh, "root:" +sDFName +":", wTempMoreParam)

                  // Copy pixel data from temporary frame into pixel data wave
                  //
                  sprintf sWave, sPixDataDecodeFormat, iInCh
                  wave pwPixData = $(sWave)

                  if(isDecoded)
                    // Decoder decoded/reconstructed the data (non loss-less)
                    //
                    Redimension/E=1/N=(dxFrDecode, dyFrDecode) wDecode
                    iImg = trunc(iPixBPerCh /(nBufPerFr *nImgPerFr))
                    pwPixData[][][iImg] = wDecode[p][q]
                    Redimension/E=1/N=(dxFrDecode *dyFrDecode) wDecode

                  else
                    // Decoder just resorted the pixel data
                    //
                    Redimension/E=1/N=(dxFrDecode *dyFrDecode *nFr *nImgPerFr) pwPixData
                    isFlipped = (mod(trunc(iPixBPerCh /nImgPerFr), 2) && (nImgPerFr > 1))
                    iBufInImg = mod(iPixBPerCh, nBufPerFr /nImgPerFr)
                    if(isFlipped)
                      m = (iPixBPerCh +1 -iBufInImg*2) *PixBLen
                      n = m +PixBLen -1
                      offsInBuf = mod(iPixBPerCh +1, nBufPerFr/nImgPerFr)*PixBLen
                    else
                      m = iPixBPerCh *PixBLen
                      n = m +PixBLen -1
                      offsInBuf = iBufInImg *PixBLen
                    endif

                    pwPixData[m,n] = wDecode[p -m +offsInBuf]
                    Redimension/E=1/N=(dxFrDecode, dyFrDecode, nFr *nImgPerFr) pwPixData
                  endif
                endif
                '''
                                iPixB += 1
                                iCh += 1

                        iPixBAllCh += 1

            # Done reading
            scm_log(f"{iPixBPerCh + 1} pixel bufs of {nPixB} read.")

            # Post-process data waves according to user settings
            isFirst = 1
            for iInCh in range(SCMIO_maxInputChans):
                if self.inputChMask & (2 ** iInCh):

                    # Reshape AI channel pixel waves
                    errC = ERR_Ok
                    if self.scanMode in [ScM_scanMode_XYImage, ScM_scanMode_XZYImage, ScM_scanMode_ZXYImage]:
                        j = 0
                        while not self._wPixData[j][0] == iInCh and j < SCMIO_maxInputChans: j += 1
                        try:
                            self._wPixData[j][1].shape = (self._nFr, int(dSlow1 / nImgPerFr), dFast)
                        except ValueError:
                            errC = ERR_CannotReshapePixelData
                    # ***************
                    # ***************
                    # TODO
                    elif self.scanMode == ScM_scanMode_TrajectArb:
                        errC = ERR_NotImplemented
                        '''
            // Just a quick fix; TODO ==>
            nFr = (nPixB/nFrPerStep*PixBLen) /(nPixPerFr) *nImgPerFr
            Redimension/E=1/N=(dFast, dSlow1/nImgPerFr, nFr) pwPixData
            '''
                    elif self.scanMode == ScM_scanMode_XYZImage:
                        errC = ERR_NotImplemented
                    # ***************
                    # ***************
                    else:
                        errC = ERR_UnknownScanCode

                    if errC != ERR_Ok:
                        s = "ERROR: " + ERRStr[errC]
                        if errC == ERR_NotImplemented:
                            s = s.format(ScM_scanModeStr[self.scanMode])
                        scm_log(s)
                        return errC

            # Save some variables for later use in properties and such
            self._dFast = dFast
            self._nFastPixRetr = nFastPixRetr
            self._nFastPixOff = nFastPixOff
            self._dSlow1 = dSlow1
            self._dSlow2 = dSlow2

            self._isSMPReady = True
            scm_log("Done.")

        except:
            raise
        return errC

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @property
    def isSMPReady(self):
        """bool: ``True`` if a pixel data file has been successfully loaded."""
        return self._isSMPReady

    def getData(self, ch=0, crop=False):
        """Return pixel data for the specified analog-input channel.

        Args:
            ch: AI channel index to retrieve (0-based).
            crop: If ``True``, the pixel-offset and retrace columns are removed,
                returning only the imaging region.

        Returns:
            numpy.ndarray or None: Array of shape ``(nFr, dSlow, dFast)`` for the
            requested channel, cropped along the fast axis when ``crop=True``.
            Returns ``None`` if the channel does not exist.
        """
        # Return data for the AIn channel `ch` or None, if channel does not exist.
        # if `crop` is True, then crop to imaging region
        for j in range(SCMIO_maxInputChans):
            if self._wPixData[j][0] == ch:
                if not crop:
                    return self._wPixData[j][1]
                else:
                    if self.scanMode in [
                        ScM_scanMode_XYImage, ScM_scanMode_XZYImage, ScM_scanMode_ZXYImage
                    ]:
                        return self._wPixData[j][1][:, :, self._nFastPixOff:-self._nFastPixRetr]
                    else:
                        assert False, "ABORT: Should not happen"
        return None

    # -------------------------------------------------------------------------------------------
