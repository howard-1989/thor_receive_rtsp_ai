dir=`pwd`

export QCAP_LOG_LEVEL=0

#for fd over1024
#export LD_PRELOAD=/home/nvidia/Documents/QtQcapMultiClientDemo_onlydecode_npptosys/select2poll/libselect2poll.so
ulimit -n 8192
#export LD_PRELOAD=/home/nvidia/Music/thor_receive_rtsp_ai/select2poll/libselect2poll.so


#export QCAP_VO_USE_GSTREAMER=1
#export QCAP_AO_LIST_PCM=1
#export QCAP_SDLMGR_VO_FPS=1

#quick startup
#export QCAP_FONT_DEFERRED_LOAD=1 

#export QCAP_DEV_VI_FPS=1
#export QCAP_DEV_AI_FPS=1
#export QCAP_VENC_FPS=1
#export QCAP_AENC_FPS=1
#export QCAP_VDEC_FPS=1
#export QCAP_ADEC_FPS=1

export LD_LIBRARY_PATH=$dir/lib:$dir/qdeep/lib
