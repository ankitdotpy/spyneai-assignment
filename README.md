# Spyne AI Assignment - Ankit Dhiman

# Setup
Clone the repository and install the dependencies:
```bash
git clone https://github.com/ankitdotpy/spyneai-assignment.git
cd spyneai-assignment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
This will install dependencies for both the projects.

## Project 1
To run the code for project 1, use the following command:
```bash
python3 project_1/main.py
```

The results are by default stores in `project_1/data/output` directory. All the requires dataset and image paths can be changed at the top of `main.py` file.

## Project 2
To run the code for project 2, use the following commands (see available options in [project_2/README.md](./project_2/README.md)):

### For training the model
```bash
./project_2/scripts/train.sh
```

### For running the api
```bash
./project_2/scripts/serve.sh
```
