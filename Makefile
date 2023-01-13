SHELL := /bin/bash

build:
	cd leafrs && maturin develop --release

clean:
	cd leafrs && rm -f Cargo.lock && cargo clean
