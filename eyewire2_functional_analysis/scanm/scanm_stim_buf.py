# ----------------------------------------------------------------------------
# scanm_stim_buf.py
# Class for ScanM stimulation buffer
#
# The MIT License (MIT)
# (c) Copyright 2022-23 Thomas Euler, Jonathan Oesterle
#
# 2022-01-31, first implementation
# 2023-06-16, changes to cope with older files
# ----------------------------------------------------------------------------
from .scanm_global import *


# ----------------------------------------------------------------------------
class StimBuf(object):
    """ Retrieves information on the stimulus buffer used for a ScanM recording
        based in the `.smh` header object
    """

    def __init__(self, smh):
        """Initialise the StimBuf object and populate it from the given SMH header.

        Args:
            smh: A loaded :class:`SMH` (or :class:`SMP`) header object from which
                stimulus-buffer parameters are read.
        """
        self._reset()
        self._smh = smh
        self._populate()

    def _reset(self):
        """ Resets object
        """
        self._isReady = False

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def _populate(self):
        """Extract stimulus-buffer parameters from the associated SMH header object.

        Reads the scan-path function name from the header to determine whether an
        external scan function was used, and sets ``pixDecodeMode`` accordingly.
        """
        nStimBuf = self._smh.nStimBuf
        lenStimBuf = self._smh.pixBufLenList[0]
        sUserScanFFull = self._smh.get(SCMIO_keys.USER_scanPathFunc)
        sUserScanFName = sUserScanFFull[0]
        self.isExtScanFunction = not (sUserScanFName in ScM_simpleScanFuncList)

        # Extract stim buffer info
        # TODO
        '''
        self.stimBufIndex =   
        self.hasCountMatrix = 
        '''
        self.pixDecodeMode = ScM_PixDataResorted  # vs. ScM_PixDataDecoded

        '''
        printf "### Looking for stimulus buffer for `%s` ...\r", sUserScanFFull
    
        // Check for stimulus buffer data folder under root, if not there, create
        // folder and inventory list
        //
        NewDataFolder/O/S $("root:" +SCMIO_stimBufFolder)
        if(!WaveExists($(SCMIO_stimBufListName)))
            Make/T/N=100  $(SCMIO_stimBufListName) = ""
        endif
        WAVE/T	pwSBList	= $(SCMIO_stimBufListName)
    
        // Check if fitting stimulus buffer is already in list
        //
        nSBList				= 0
        isSBFound			= 0
        do
            isEmpty			= strlen(pwSBList[nSBList]) == 0
            if(!isEmpty)
                isSBFound	= StringMatch(pwSBList[nSBList], sUserScanFFull)
                if(isSBFound)
                    // Matching stimulus buffer entry found, return reference
                    // to stimulus buffer wave
                    //
                    printf "### Matching stimulus buffer already exists\r"
    
                    pwInfo[%stimBufIndex]		= nSBList
                    pwInfo[%hasCountMatrix]	= WaveExists($(SCMIO_scanDecCountMatrixName +"_" +Num2Str(nSBList)))
                    WAVE 	pwOther		= $(SCMIO_otherParamsWaveName +"_" +Num2Str(nSBList))
                    pwInfo[%pixDecodeMode]		= pwOther[%pixDecodeMode]
    
                    SetDataFolder saveDFR
                    return pwInfo
                else
                    nSBList		+= 1
                endif
            endif
        while(!isEmpty)
        printf "### No matching stimulus buffer found, recreating ...\r"
    
        // Create stimulus buffer wave ...
        //
        sTemp	= "wStimBufData_" +Num2Str(nSBList)
        Make/O/S/N=(nStimBuf, lenStimBuf) $(sTemp)
        WAVE	pwStimBufData	= $(sTemp)
    
        // Fill stimulus buffer with scan path data
        //
        switch(pwPNum[%User_ScanMode])
            case ScM_scanMode_XYImage		:
            case ScM_scanMode_XZYImage	:
            case ScM_scanMode_ZXYImage	:
                AOCh3Scale	= pwPNum[%MaxVolts_AO]
                break
    
            case ScM_scanMode_TrajectArb	:
                // Other than for "standard" scans, here no scaling (i.e. for the blanking
                // signal) happens; it is expected that the stimuli contain final voltages
                //
                AOCh3Scale	= 1.0
                break
        endswitch
    
        // Call scan path function and receive four buffers (see below) with the scan path
        // data for one frame
        //
        ScM_callScanPathFunc(sUserScanFFull)
        if(!WaveExists($("StimX")))
            print "### Something went wrong, please retry ..."
            return $("StimX")
        endif
        wave pwTempX				= $("StimX")
        wave pwTempY				= $("StimY")
        wave pwTempPC				= $("StimPC")
        wave pwTempZ				= $("StimZ")
        wave pwScanPathFuncParams	= $("wScanPathFuncParams")
        sTemp						= "wScanPathFuncParams_" +Num2Str(nSBList)
        Duplicate/O pwScanPathFuncParams, $(sTemp)
        KillWaves/Z pwScanPathFuncParams
        wave pwScanPathFuncParams	= $(sTemp)
    
        sTemp	= SCMIO_otherParamsWaveName +"_" +Num2Str(nSBList)
        Make/O/N=(1) $(sTemp)
        WAVE 	pwOther		= $(sTemp)
        SetDimLabel 0, 0, 	pixDecodeMode,		pwOther
    
        // Copy scan path data into the stimulus buffer
        //
        for(j=0; j<lenStimBuf; j+=1)
            pwStimBufData[SCM_indexScannerX][j]	= pwTempX[q]
            pwStimBufData[SCM_indexScannerY][j]	= pwTempY[q]
        //	pwStimBufData[SCM_indexLaserBlk][j]	= pwTempPC[q] *AOCh3Scale
             pwStimBufData[SCM_indexLaserBlk][j]	= (pwTempPC[q] > 0)?(ScM_TTLhigh):(ScM_TTLlow)
            pwStimBufData[SCM_indexLensZ][j]		= pwTempZ[q] *pwPNum[%User_ETL_polarity_V]
        endfor
    
        if(isExtScanF)
            // If an external scan path function is defined, prepare its decoder
            //
            FUNCREF ScM_ScanPathPrepDecodeProtoFunc fPrepDecode	= $(sUserScanFName + "_prepareDecode")
            result 	= fPrepDecode(pwStimBufData, pwScanPathFuncParams)
            pwInfo[%pixDecodeMode]		= result
            pwOther[%pixDecodeMode]	= pwInfo[%pixDecodeMode]
    
            if(WaveExists(countMatrix))
                sTemp	= SCMIO_scanDecCountMatrixName +"_" +Num2Str(nSBList)
                Duplicate/O $(SCMIO_scanDecCountMatrixName), $(sTemp)
                KillWaves/Z $(SCMIO_scanDecCountMatrixName)
            endif
        endif
    
        // Save entry in stimulus buffer list and fill out info wave
        //
        pwSBList[nSBList]			= sUserScanFFull
        pwInfo[%stimBufIndex]		= nSBList
        pwInfo[%hasCountMatrix]	= WaveExists($(SCMIO_scanDecCountMatrixName +"_" +Num2Str(nSBList)))
    
        // Clean up
        //
        KillWaves/Z pwTempX, pwTempY, pwTempPC, pwTempZ, wStimBufData_temp
        SetDataFolder saveDFR
    
        return pwInfo
        '''

    # ----------------------------------------------------------------------------
