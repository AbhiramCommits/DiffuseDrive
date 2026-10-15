.PHONY: install lint test fixtures train-small eval-all plan-all report reproduce clean

install:
	pip install -e .[dev]

lint:
	ruff check .

test:
	pytest -q --cov=diffusedrive

fixtures:
	python scripts/make_fixtures.py

train-small:
	python -m diffusedrive.train.cli --config configs/small_cpu.yaml --model diffusion --out runs/diffusion
	python -m diffusedrive.train.cli --config configs/small_cpu.yaml --model mixture --out runs/mixture
	python -m diffusedrive.train.cli --config configs/small_cpu.yaml --model bc --out runs/bc

eval-all:
	python -m diffusedrive.eval.cli --ckpt runs/diffusion/ckpt_best.pt --K 6 --out results/diffusion_k6.json
	python -m diffusedrive.eval.cli --ckpt runs/mixture/ckpt_best.pt --K 6 --out results/mixture_k6.json
	python -m diffusedrive.eval.cli --ckpt runs/bc/ckpt_best.pt --K 1 --out results/bc_k1.json

plan-all:
	python -m diffusedrive.plan.cli --planner cv --episodes 20 --out results/closed_loop_cv.json
	python -m diffusedrive.plan.cli --planner bc --ckpt runs/bc/ckpt_best.pt --episodes 20 --out results/closed_loop_bc.json
	python -m diffusedrive.plan.cli --planner diffusion --ckpt runs/diffusion/ckpt_best.pt --episodes 20 --out results/closed_loop_diffusion.json

report:
	python -m diffusedrive.report

reproduce: clean install fixtures test train-small eval-all plan-all report

clean:
	rm -rf runs results __pycache__ .pytest_cache .coverage
