%module(directors="1") AoceWrapper
%feature("director") IMediaPlayerObserver;
%{   
#include <aoce/Aoce.h>
#include <aoce/Layer/BaseLayer.hpp>
#include <aoce/AoceManager.hpp>
#include <aoce/VideoDevice/VideoManager.hpp>
#include <aoce/Media/MediaPlayer.hpp>
%}

%rename(LoadAoce) loadAoce;
%rename(UnloadAoce) unloadAoce;

// Import standard types.
%include "std_vector.i"
%include "stdint.i"

%feature("director") LogEventHandle;
%inline %{
    class LogEventHandle {
    public:
        LogEventHandle() = default;
        virtual ~LogEventHandle() = default;       
        virtual void onLogEvent(int level, const char* message) = 0;        
        logEventHandle createWrapper() {
            return [this](int level, const char* message) -> void {
                onLogEvent(level, message);
            };
        }
    };
%}

%feature("director") IMediaPlayerObserver;

%inline %{
void SetLogHandle(LogEventHandle* handle){
    setLogHandle(handle->createWrapper());
}
%}

void loadAoce();

void unloadAoce();

%nodefaultctor ;
%nodefaultdtor ;

namespace aoce {
class VideoDevice{
    public:
     bool open();
};

// class MediaPlayer{
//     public:
//     void setObserver(IMediaPlayerObserver* observer);
// };
}

%clearnodefaultctor; 
%clearnodefaultdtor; 

#include <aoce/Aoce.h>
