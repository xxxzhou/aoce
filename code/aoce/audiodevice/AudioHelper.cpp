#include <vector>

#include "AudioDevice.hpp"

namespace aoce {
//只用来写formatTag为固定的WAVE_FORMAT_PCM结构
struct WaveFormat {
    //写文件,在这固定为PCM格式
    uint16_t formatTag = 1;
    uint16_t channel = 1;
    uint32_t sampleRate = 44100;
    uint32_t sampleBytes = 88200;
    uint16_t blockAlign = 2;
    uint16_t bitSize = 16;  // 16U
    uint16_t cbSize = 0;
};

struct WAVEHEADER {
    uint32_t dwRiff;     // "RIFF"
    uint32_t dwSize;     // Size
    uint32_t dwWave;     // "WAVE"
    uint32_t dwFmt;      // "fmt "
    uint32_t dwFmtSize;  // Wave Format Size
};

// Static RIFF header, we'll append the format to it.
const uint8_t WaveHeader[] = {'R',  'I', 'F',  'F',  0x00, 0x00, 0x00,
                              0x00, 'W', 'A',  'V',  'E',  'f',  'm',
                              't',  ' ', 0x00, 0x00, 0x00, 0x00};

const uint8_t WaveData[] = {'d', 'a', 't', 'a'};

void getWavHeader(std::vector<uint8_t>& header, uint32_t dataSize,
                  const AudioFormat& audioDesc) {
    uint32_t headerSize = sizeof(WAVEHEADER) + sizeof(WaveFormat) +
                          sizeof(WaveData) + sizeof(uint32_t);
    header.resize(headerSize, 0);
    
    WaveFormat format = {};
    format.channel = audioDesc.channel;
    format.sampleRate = audioDesc.sampleRate;
    format.bitSize = audioDesc.depth;
    format.blockAlign = format.channel * format.bitSize / 8;
    format.sampleBytes = format.sampleRate * format.blockAlign;

    uint8_t* writeIdx = header.data();
    // 先写声音基本信息
    WAVEHEADER* waveHeader = reinterpret_cast<WAVEHEADER*>(writeIdx);
    memcpy(writeIdx, WaveHeader, sizeof(WAVEHEADER));
    writeIdx += sizeof(WAVEHEADER);
    waveHeader->dwSize = headerSize + dataSize - 2 * 4;
    waveHeader->dwFmtSize = sizeof(WaveFormat);
    // format后面可能会带cbSize个额外信息
    memcpy(writeIdx, &format, sizeof(WaveFormat));
    writeIdx += sizeof(WaveFormat);
    //写入data
    memcpy(writeIdx, WaveData, sizeof(WaveData));
    writeIdx += sizeof(WaveData);
    //写入一个结尾数据长度
    *(reinterpret_cast<uint32_t*>(writeIdx)) = static_cast<uint32_t>(dataSize);
    writeIdx += sizeof(uint32_t);
}

}  // namespace aoce