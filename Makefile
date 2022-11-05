SHELL := /bin/bash

build:
	python3 -m venv venv && source venv/bin/activate
	python3 -m pip install -r requirements.txt
	cd leafrs && maturin develop --release

clean:
	cd leafrs && rm -f Cargo.lock && cargo clean

