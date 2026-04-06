-----------------
Jason Yoo
Apr 6, 2026
-----------------

# General
We will use virtual environment (venv). This ensures that we work on the same environment and get the same results. Without sharing the same python libary _files_, we will share the _list_ of libraries that we agreed to use.

You only need to set up once. But you will need to activate the virtual environment every time.

# Venv setup
Move to the working directory
```bash
cd ./final_project
```

Run the command to initialize the virtual environment
```bash
python -m venv .venv
```

Install all the dependencies (packages that we all agreed to use)
```bash
pip install -r requirements.txt
```

If you ever decide to install more packages, please install and update the requirement file by the command below
```bash
pip freeze > requirements.txt
```
And then please let others know we have a package to install!

## Jupyter Notebook on VS code
Also, when you are using Jupyter Notebook, you might need to do extra setup.

1. Pres Ctrl+Shift+P.
2. Search `Python: select interpreter`
3. Select `Etner interpreter path`
4. Find and select `./.venv/Scripts/python.exe`


# Venv Activate
You should activate the environment every time you run a code. If you don't, although your code may still work, your result may be different from what others see.

Move to the working directory
```bash
cd ./final_project
```

Use the following commands to actiavate
```bash
source .venv/Scripts/activate
```

To deactiavate, you can either use the following commands or just close the terminal.
```bash
source .venv/Scripts/deactiavte
```

# Config setup
Since we don't have the dataset on the github repo, we will most likely have the dataset in a different path. In the shared files, we don't want to hardcode the path, so we will use `config.json`.

Open your `config.json`, and replace the placeholder with your actual dataset path.
```json
{
  "dataset_path": "<Change THIS>"
}
```
We will update this `config.json` when we need to use anything user-specific but also need to share.

How you load this config and load the dataset is shown in `load_dataset.ipynb`. 