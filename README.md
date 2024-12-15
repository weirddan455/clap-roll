# clap-roll
CLAP Audio Plugin for 1000% better sound in DOS games

## Installation
Linux and Windows binaries are available here: https://github.com/weirddan455/clap-roll/releases

DOSBox Staging CLAP support is available in this PR: https://github.com/dosbox-staging/dosbox-staging/pull/4090

Copy the .clap file into the DOSBox Staging Plugins directory
Linux: ~/.config/dosbox/plugins
Windows: \AppData\Local\Dosbox\plugins

Set mididevice = soundcanvas in dosbox-staging.conf

## Build Instructions
Install the Rust toolchain using instructions here: https://www.rust-lang.org/learn/get-started
It's a very easy one-click install

Run cargo build --release from the command line
Rename the .dll (Windows) or .so (Linux) inside target/release file to .clap
Follow install instructions above
