use std::alloc::{alloc, dealloc, Layout};
use std::borrow::Cow;
use std::ffi::{c_char, c_float, c_double, c_void, CStr};
use std::io::Read;
use std::ops::DerefMut;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::ptr;

use hex_literal::hex;

use rubato::{FftFixedInOut, Resampler};

use sha2::{Digest, Sha256};

use symphonia::core::audio::{AudioBuffer, AudioBufferRef, Signal};
use symphonia::core::codecs::{Decoder, DecoderOptions};
use symphonia::core::formats::{FormatReader, FormatOptions, SeekMode, SeekTo};
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

#[repr(C)]
struct ClapVersion {
    major: u32,
    minor: u32,
    revision: u32
}

const CLAP_VERSION: ClapVersion = ClapVersion {
    major: 1,
    minor: 2,
    revision: 2
};

#[repr(C)]
struct ClapPluginEntry {
    clap_version: ClapVersion,
    init: extern "C" fn(*const c_char) -> bool,
    deinit: extern "C" fn(),
    get_factory: extern "C" fn(*const c_char) -> *const ClapPluginFactory
}

#[no_mangle]
static clap_entry: ClapPluginEntry = ClapPluginEntry {
    clap_version: CLAP_VERSION,
    init: entry_init,
    deinit: entry_deinit,
    get_factory: get_factory
};

extern "C" fn entry_init(_plugin_path: *const c_char) -> bool {
    true
}

extern "C" fn entry_deinit() {
}

extern "C" fn get_factory(_factory_id: *const c_char) -> *const ClapPluginFactory {
    &PLUGIN_FACTORY
}

#[repr(C)]
struct ClapPluginFactory {
    get_plugin_count: extern "C" fn(*const ClapPluginFactory) -> u32,
    get_plugin_descriptor: extern "C" fn(*const ClapPluginFactory, u32) -> *const ClapPluginDescriptor,
    create_plugin: extern "C" fn(*const ClapPluginFactory, *const ClapHost, *const c_char) -> *const ClapPlugin
}

const PLUGIN_FACTORY: ClapPluginFactory = ClapPluginFactory {
    get_plugin_count: get_plugin_count,
    get_plugin_descriptor: get_plugin_descriptor,
    create_plugin: create_plugin
};

extern "C" fn get_plugin_count(_factory: *const ClapPluginFactory) -> u32 {
    1
}

extern "C" fn get_plugin_descriptor(_factory: *const ClapPluginFactory, _index: u32) -> *const ClapPluginDescriptor {
    &PLUGIN_DESCRIPTOR
}

extern "C" fn create_plugin(_factory: *const ClapPluginFactory, _host: *const ClapHost, _plugin_id: *const c_char) -> *const ClapPlugin {
    unsafe {
        let plugin = alloc(Layout::new::<ClapPlugin>()) as *mut ClapPlugin;
        if plugin.is_null() {
            return ptr::null();
        }
        let meme_stream = alloc(Layout::new::<Mutex<MemeStream>>()) as *mut Mutex<MemeStream>;
        if meme_stream.is_null() {
            dealloc(plugin as *mut u8, Layout::new::<ClapPlugin>());
            return ptr::null();
        }
        meme_stream.write(Mutex::new(MemeStream::new()));
        plugin.write(ClapPlugin{
            desc: &PLUGIN_DESCRIPTOR,
            meme_stream,
            init: plugin_init,
            destroy: plugin_destroy,
            activate: plugin_activate,
            deactivate: plugin_deactivate,
            start_processing: start_processing,
            stop_processing: stop_processing,
            reset: plugin_reset,
            process: process_audio,
            get_extension: get_extension,
            on_main_thread: on_main_thread
        });
        plugin
    }
}

#[repr(C)]
struct ClapPluginDescriptor {
    clap_version: ClapVersion,
    id: *const c_char,
    name: *const c_char,
    vendor: *const c_char,
    url: *const c_char,
    manual_url: *const c_char,
    support_url: *const c_char,
    version: *const c_char,
    description: *const c_char,
    features: *const *const c_char
}

const PLUGIN_DESCRIPTOR: ClapPluginDescriptor = ClapPluginDescriptor {
    clap_version: CLAP_VERSION,
    id: c"com.weirddan455.clap-roll".as_ptr(),
    name: c"Roland SC-55 v1.21".as_ptr(),
    vendor: c"weirddan455".as_ptr(),
    url: c"github.com".as_ptr(),
    manual_url: c"github.com".as_ptr(),
    support_url: c"github.com".as_ptr(),
    version: c"0.69".as_ptr(),
    description: c"The dankest of memes".as_ptr(),
    features: [c"dank".as_ptr(), c"memes".as_ptr(), ptr::null()].as_ptr()
};

#[repr(C)]
struct ClapEventHeader {
    size: u32,
    time: u32,
    space_id: u16,
    event_type: u16,
    flags: u32
}

#[repr(C)]
struct ClapEventTransport {
    header: ClapEventHeader,
    flags: u32,
    song_pos_beats: i64,
    song_pos_seconds: i64,
    tempo: c_double,
    temp_inc: c_double,
    loop_start_beats: i64,
    loop_end_beats: i64,
    loop_start_seconds: i64,
    loop_end_seconds: i64,
    bar_start: i64,
    bar_number: i64,
    tsig_num: u16,
    tsig_denom: u16
}

#[repr(C)]
struct ClapAudioBuffer {
    data32: *mut *mut c_float,
    data64: *mut *mut c_double,
    channel_count: u32,
    latency: u32,
    constant_mask: u64
}

#[repr(C)]
struct ClapInputEvents {
    ctx: *mut c_void,
    size: extern "C" fn(*const ClapInputEvents) -> u32,
    get: extern "C" fn(*const ClapInputEvents, u32) -> *const ClapEventHeader
}

#[repr(C)]
struct ClapOutputEvents {
    ctx: *mut c_void,
    try_push: extern "C" fn(*const ClapOutputEvents, *const ClapEventHeader) -> bool
}

#[repr(C)]
struct ClapProcess {
    steady_time: i64,
    frames_count: u32,
    transport: *const ClapEventTransport,
    audio_inputs: *const ClapAudioBuffer,
    audio_outputs: *mut ClapAudioBuffer,
    audio_inputs_count: u32,
    audio_outputs_count: u32,
    in_events: *const ClapInputEvents,
    out_events: *const ClapOutputEvents
}

enum MemeStreamState {
    NeedsInit,
    FailedToInit,
    GoodToGo
}

struct MemeStream {
    state: MemeStreamState,
    format_reader: Option<Box<dyn FormatReader>>,
    decoder: Option<Box<dyn Decoder>>,
    resampler: Option<FftFixedInOut<f32>>,
    sample_rate_out: usize,
    output_buffer_index: usize,
    output_buffer_left: Vec<f32>,
    output_buffer_right: Vec<f32>,
    resample_buffer_index: usize,
    resample_buffer_left: Vec<f32>,
    resample_buffer_right: Vec<f32>
}

impl MemeStream {
    fn new() -> Self {
        Self {
            state: MemeStreamState::NeedsInit,
            format_reader: None,
            decoder: None,
            resampler: None,
            sample_rate_out: 0,
            output_buffer_index: 0,
            output_buffer_left: Vec::new(),
            output_buffer_right: Vec::new(),
            resample_buffer_index: 0,
            resample_buffer_left: Vec::new(),
            resample_buffer_right: Vec::new()
        }
    }
}

#[repr(C)]
struct ClapPlugin {
    desc: *const ClapPluginDescriptor,
    meme_stream: *const Mutex<MemeStream>,
    init: extern "C" fn(*const ClapPlugin) -> bool,
    destroy: extern "C" fn(*const ClapPlugin),
    activate: extern "C" fn(*const ClapPlugin, c_double, u32, u32) -> bool,
    deactivate: extern "C" fn(*const ClapPlugin), 
    start_processing: extern "C" fn(*const ClapPlugin) -> bool,
    stop_processing: extern "C" fn(*const ClapPlugin),
    reset: extern "C" fn(*const ClapPlugin),
    process: extern "C" fn(*const ClapPlugin, *const ClapProcess) -> i32,
    get_extension: extern "C" fn(*const ClapPlugin, *const c_char) -> *const c_void,
    on_main_thread: extern "C" fn(*const ClapPlugin)
}

extern "C" fn plugin_init(_plugin: *const ClapPlugin) -> bool {
    true
}

extern "C" fn plugin_destroy(plugin: *const ClapPlugin) {
    unsafe {
        std::ptr::drop_in_place((*plugin).meme_stream as *mut Mutex<MemeStream>);
        dealloc((*plugin).meme_stream as *mut u8, Layout::new::<Mutex<MemeStream>>());
        dealloc(plugin as *mut u8, Layout::new::<ClapPlugin>());
    }
}

extern "C" fn plugin_activate(plugin_ptr: *const ClapPlugin, sample_rate: c_double, _min_frames_count: u32, _max_frames_count: u32) -> bool {
    static SPAWN_THREAD: AtomicBool = AtomicBool::new(true);
    if SPAWN_THREAD.fetch_and(false, Ordering::Relaxed) {
        std::thread::spawn(get_meme);
    }
    let plugin = unsafe{&*plugin_ptr};
    let mutex = unsafe{&*plugin.meme_stream};
    let mut guard = match mutex.lock() {
        Ok(g) => g,
        Err(_) => return false
    };
    let stream = guard.deref_mut();
    let requested_sample_rate = sample_rate.round() as usize;
    if stream.sample_rate_out != requested_sample_rate {
        stream.sample_rate_out = requested_sample_rate;
        stream.resampler = if requested_sample_rate == 44100 {
            None
        } else {
            match FftFixedInOut::<f32>::new(44100, requested_sample_rate, 1024, 2) {
                Ok(r) => Some(r),
                Err(_) => None
            }
        }
    }
    true
}

extern "C" fn plugin_deactivate(_plugin: *const ClapPlugin) {
}

extern "C" fn start_processing(_plugin: *const ClapPlugin) -> bool {
    true
}

extern "C" fn stop_processing(_plugin: *const ClapPlugin) {
}

extern "C" fn plugin_reset(_plugin: *const ClapPlugin) {
}

fn decode_audio<'a>(decoder: &'a mut dyn Decoder, format_reader: &mut dyn FormatReader) -> Option<Cow<'a, AudioBuffer<f32>>> {
    let packet = match format_reader.next_packet() {
        Ok(p) => p,
        Err(_) => {
            let track_id = match format_reader.default_track() {
                Some(t) => t.id,
                None => 0
            };
            let _ = format_reader.seek(SeekMode::Accurate, SeekTo::TimeStamp{ts: 0, track_id});
            match format_reader.next_packet() {
                Ok(p) => {
                    decoder.reset();
                    p
                },
                Err(_) => return None
            }
        }
    };

    match decoder.decode(&packet) {
        Ok(AudioBufferRef::F32(b)) => Some(b),
        _ => None
    }
}

extern "C" fn process_audio(plugin: *const ClapPlugin, process_ptr: *const ClapProcess) -> i32 {
    const CLAP_ERR: i32 = 0;
    const CLAP_OK: i32 = 1;
    let encoded_data = match DANK_MEMES.get() {
        Some(s) => s,
        None => return CLAP_OK
    };
    let mutex = unsafe{&*(*plugin).meme_stream};
    let mut guard = match mutex.lock() {
        Ok(g) => g,
        Err(_) => return CLAP_ERR
    };
    let stream = guard.deref_mut();
    match stream.state {
        MemeStreamState::FailedToInit => return CLAP_ERR,
        MemeStreamState::NeedsInit => {
            stream.state = MemeStreamState::FailedToInit;
            let cursor = std::io::Cursor::new(encoded_data);
            let source = MediaSourceStream::new(Box::new(cursor), MediaSourceStreamOptions::default());
            let format_reader = match symphonia::default::get_probe().format(Hint::default().with_extension("ogg"), source, &FormatOptions::default(), &MetadataOptions::default()) {
                Ok(p) => p.format,
                Err(_) => return CLAP_ERR
            };
            let track = match format_reader.default_track() {
                Some(t) => t,
                None => return CLAP_ERR
            };
            let decoder = match symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default()) {
                Ok(d) => d,
                Err(_) => return CLAP_ERR
            };
            stream.format_reader = Some(format_reader);
            stream.decoder = Some(decoder);
            stream.state = MemeStreamState::GoodToGo;
        },
        MemeStreamState::GoodToGo => {}
    }
    let format_reader = match &mut stream.format_reader {
        Some(r) => r.as_mut(),
        None => return CLAP_ERR
    };
    let decoder = match &mut stream.decoder {
        Some(d) => d.as_mut(),
        None => return CLAP_ERR
    };
    let process = unsafe{&*process_ptr};
    let frame_count = process.frames_count as usize;
    let outputs = unsafe{&*process.audio_outputs};
    let channels = unsafe{std::slice::from_raw_parts(outputs.data32, outputs.channel_count as usize)};
    let dst_left = unsafe{std::slice::from_raw_parts_mut(channels[0], frame_count)};
    let dst_right = unsafe{std::slice::from_raw_parts_mut(channels[1], frame_count)};

    let mut dst_i = if stream.output_buffer_index < stream.output_buffer_left.len() {
        let frames = frame_count.min(stream.output_buffer_left.len() - stream.output_buffer_index);
        dst_left[0..frames].copy_from_slice(&stream.output_buffer_left[stream.output_buffer_index..stream.output_buffer_index + frames]);
        dst_right[0..frames].copy_from_slice(&stream.output_buffer_right[stream.output_buffer_index..stream.output_buffer_index + frames]);
        stream.output_buffer_index += frames;
        frames
    } else {
        0
    };

    if let Some(resampler) = &mut stream.resampler {
        while dst_i < frame_count {
            let to_resample = stream.resample_buffer_left.len() - stream.resample_buffer_index;
            let input_frames = resampler.input_frames_next();
            if to_resample >= input_frames {
                let in_left = &stream.resample_buffer_left[stream.resample_buffer_index..stream.resample_buffer_index + input_frames];
                let in_right = &stream.resample_buffer_right[stream.resample_buffer_index..stream.resample_buffer_index + input_frames];
                let resample_in = [in_left, in_right];
                let remaining = frame_count - dst_i;
                let output_frames = resampler.output_frames_next();
                if remaining >= output_frames {
                    let out_left = &mut dst_left[dst_i..dst_i + output_frames];
                    let out_right = &mut dst_right[dst_i..dst_i + output_frames];
                    let mut out = [out_left, out_right];
                    let _ = resampler.process_into_buffer(&resample_in, &mut out, None);
                    dst_i += output_frames;
                    stream.resample_buffer_index += input_frames;
                    continue;
                } else {
                    stream.output_buffer_left.resize(output_frames, 0.0);
                    stream.output_buffer_right.resize(output_frames, 0.0);
                    let mut out = [&mut stream.output_buffer_left, &mut stream.output_buffer_right];
                    let _ = resampler.process_into_buffer(&resample_in, &mut out, None);
                    dst_left[dst_i..].copy_from_slice(&stream.output_buffer_left[0..remaining]);
                    dst_right[dst_i..].copy_from_slice(&stream.output_buffer_right[0..remaining]);
                    stream.output_buffer_index = remaining;
                    stream.resample_buffer_index += input_frames;
                    return CLAP_OK;
                }
            }
            let decoded_buffer = match decode_audio(decoder, format_reader) {
                Some(b) => b,
                None => return CLAP_ERR
            };
            let src_left = decoded_buffer.chan(0);
            let src_right = decoded_buffer.chan(1);
            assert_eq!(src_left.len(), src_right.len());
            stream.resample_buffer_left.rotate_left(stream.resample_buffer_index);
            stream.resample_buffer_left.truncate(to_resample);
            stream.resample_buffer_left.extend_from_slice(src_left);
            stream.resample_buffer_right.rotate_left(stream.resample_buffer_index);
            stream.resample_buffer_right.truncate(to_resample);
            stream.resample_buffer_right.extend_from_slice(src_right);
            stream.resample_buffer_index = 0;
        }
    } else {
        while dst_i < frame_count {
            let decoded_buffer = match decode_audio(decoder, format_reader) {
                Some(b) => b,
                None => return CLAP_ERR
            };
            let src_left = decoded_buffer.chan(0);
            let src_right = decoded_buffer.chan(1);
            assert_eq!(src_left.len(), src_right.len());
            let remaining = frame_count - dst_i;
            let frames = remaining.min(src_left.len());
            dst_left[dst_i..dst_i + frames].copy_from_slice(&src_left[0..frames]);
            dst_right[dst_i..dst_i + frames].copy_from_slice(&src_right[0..frames]);
            dst_i += frames;
            if dst_i == frame_count {
                stream.output_buffer_left.clear();
                stream.output_buffer_left.extend_from_slice(&src_left[frames..]);
                stream.output_buffer_right.clear();
                stream.output_buffer_right.extend_from_slice(&src_right[frames..]);
                stream.output_buffer_index = 0;
            }
        }
    }

    CLAP_OK
}

extern "C" fn get_extension(_plugin: *const ClapPlugin, id: *const c_char) -> *const c_void {
    let cstr = unsafe{CStr::from_ptr(id)};
    if cstr == c"clap.note-ports" {
        return &NOTE_PORTS as *const _ as *const c_void;
    } else if cstr == c"clap.audio-ports" {
        return &AUDIO_PORTS as *const _ as *const c_void;
    }
    ptr::null()
}

extern "C" fn on_main_thread(_plugin: *const ClapPlugin) {
}

#[repr(C)]
struct ClapHost {
    clap_version: ClapVersion,
    host_data: *mut c_void,
    name: *const c_char,
    vendor: *const c_char,
    url: *const c_char,
    version: *const c_char,
    get_extension: extern "C" fn(*const ClapHost, *const c_char) -> *const c_void,
    request_restart: extern "C" fn(*const ClapHost),
    request_process: extern "C" fn(*const ClapHost),
    request_callback: extern "C" fn(*const ClapHost),
}

#[repr(C)]
struct ClapNotePortInfo {
    id: u32,
    supported_dialects: u32,
    preferred_dialect: u32,
    name: [c_char; 256]
}

#[repr(C)]
struct ClapPluginNotePorts {
    count: extern "C" fn(*const ClapPlugin, bool) -> u32,
    get: extern "C" fn(*const ClapPlugin, u32, bool, *mut ClapNotePortInfo) -> bool
}

const NOTE_PORTS: ClapPluginNotePorts = ClapPluginNotePorts {
    count: note_ports_count,
    get: get_note_port_info
};

extern "C" fn note_ports_count(_plugin: *const ClapPlugin, is_input: bool) -> u32 {
    if is_input {1} else {0}
}

extern "C" fn get_note_port_info(_plugin: *const ClapPlugin, index: u32, is_input: bool, port_info: *mut ClapNotePortInfo) -> bool {
    if !is_input || index > 0 {
        return false;
    }
    let mut info = ClapNotePortInfo {
        id: 0,
        supported_dialects: 2,
        preferred_dialect: 2,
        name: [0; 256]
    };
    copy_to_cstring_array(&mut info.name, b"Note Port");
    unsafe {
        port_info.write(info);
    }
    true
}

#[repr(C)]
struct ClapAudioPortInfo {
    id: u32,
    name: [c_char; 256],
    flags: u32,
    channel_count: u32,
    port_type: *const c_char,
    in_place_pair: u32
}

#[repr(C)]
struct ClapPluginAudioPorts {
    count: extern "C" fn(*const ClapPlugin, bool) -> u32,
    get: extern "C" fn(*const ClapPlugin, u32, bool, *mut ClapAudioPortInfo) -> bool
}

const AUDIO_PORTS: ClapPluginAudioPorts = ClapPluginAudioPorts {
    count: audio_port_count,
    get: get_audio_port_info
};

extern "C" fn audio_port_count(_plugin: *const ClapPlugin, is_input: bool) -> u32 {
    if is_input {0} else {1}
}

extern "C" fn get_audio_port_info(_plugin: *const ClapPlugin, index: u32, is_input: bool, port_info: *mut ClapAudioPortInfo) -> bool {
    if is_input || index > 0 {
        return false;
    }
    let mut info = ClapAudioPortInfo {
        id: 0,
        name: [0; 256],
        flags: 1,
        channel_count: 2,
        port_type: c"stereo".as_ptr(),
        in_place_pair: u32::MAX
    };
    copy_to_cstring_array(&mut info.name, b"Audio Port");
    unsafe {
        port_info.write(info);
    }
    true
}

fn copy_to_cstring_array(dst: &mut [c_char], src: &[u8]) {
    if dst.len() == 0 {
        return;
    }
    let mut i = 0;
    for c in src {
        if i >= dst.len() - 1 {
            break;
        }
        dst[i] = *c as c_char;
        i += 1;
    }
    dst[i] = 0;
}

static DANK_MEMES: OnceLock<Vec<u8>> = OnceLock::new();

fn get_meme() {
    const GOOD_MEME: [u8; 32] = hex!("d9e742e4582b0516bd4152b6f06cb13b5095b8f550c011431a601ecc3b74df77");
    let meme_path = match dirs::cache_dir() {
        Some(mut p) => {
            p.push("clap-roll");
            p.push("rickroll.ogg");
            if let Ok(meme) = std::fs::read(&p) {
                if Sha256::digest(&meme)[..] == GOOD_MEME {
                    let _ = DANK_MEMES.set(meme);
                    return;
                }
            }
            Some(p)
        },
        None => None
    };

    if let Ok(response) = ureq::get("https://archive.org/download/NeverGonnaGiveYouUp/jocofullinterview41.ogg").call() {
        let mut meme = Vec::new();
        if let Ok(_) = response.into_reader().take(10_000_000).read_to_end(&mut meme) {
            if Sha256::digest(&meme)[..] == GOOD_MEME {
                let _ = DANK_MEMES.set(meme);
                if let Some(p) = meme_path {
                    if let Some(cached) = DANK_MEMES.get() {
                        if let Some(parent) = p.parent() {
                            let _ = std::fs::create_dir(parent);
                            let _ = std::fs::write(p, cached);
                        }
                    }
                }
            }
        }
    }
}
