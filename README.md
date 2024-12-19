# DeepNN

DeepNN is a deep learning framework designed for Professor Xiao Liu's class. This guide provides step-by-step instructions to help you set up and run DeepNN on your local machine.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Create and Activate a Virtual Environment](#create-and-activate-a-virtual-environment)
- [Clone the Repository](#clone-the-repository)
- [Install Necessary Python Packages](#install-necessary-python-packages)
- [Run the Pipeline](#run-the-pipeline)
  - [Update and Run the `run.sh` Script](#update-and-run-the-runsh-script)
  - [Alternative: Run the `pipeline.sh` Script](#alternative-run-the-pipelinesh-script)
- [Re-run the Program](#re-run-the-program)

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- [Anaconda](https://www.anaconda.com/products/distribution) (for managing environments)
- [Git](https://git-scm.com/downloads) (for cloning the repository)
- [Python 3.8](https://www.python.org/downloads/release/python-380/)
- [R](https://www.r-project.org/) (required for the `r-data.table` package)

- **Note:** Please make sure that these are updated to the newest version

## Create and Activate a Virtual Environment

1. **Open your terminal or command prompt.**

2. **Create a new Conda environment named `deepnn` with Python 3.8:**

   ```bash
   conda create --name deepnn python=3.8
   ```

3. **Activate the `deepnn` environment:**

   ```bash
   conda activate deepnn
   ```

## Clone the Repository

1. **Navigate to the directory where you want to clone the repository:**

   ```bash
   cd /path/to/your/preferred/directory
   ```

2. **Clone the DeepNN repository:**

   ```bash
   git clone https://github.com/sichenz/DeepNN.git
   ```

3. **Navigate into the cloned repository:**

   ```bash
   cd DeepNN
   ```

## Install Necessary Python Packages

1. **Upgrade `pip`, `setuptools`, and `wheel`:**

   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. **Run the setup script to install libraries:**

   ```bash
   bash setup/run_setup.sh
   ```

   - **Note:** Ensure that the `run_setup.sh` script has execute permissions. If not, you can make it executable with:

     ```bash
     chmod +x setup/run_setup.sh
     ```

3. **Run the setup script to install R packages:**

   ```bash
   Rscript setup/setup_packages.R
   ```

## Run the Pipeline

### Update and Run the `run.sh` Script

1. **Open the `run.sh` script in a text editor or on VSCode.**

2. **Locate the `cd` command in the script and update it to point to the absolute path of your cloned `DeepNN` repository.**

   - **Example:**

     ```bash
     cd /Users/yourusername/path/to/DeepNN
     ```

3. **Save and close the `run.sh` script.**

4. **Make sure the `run.sh` script has execute permissions:**

   ```bash
   chmod +x run.sh
   ```

5. **Run the pipeline script to generate log script for data retrieval:**

   ```bash
   ./run.sh
   ```

  - **WARNING:** The code will take around TWO HOURS to run - THIS IS NORMAL!

### Alternative: Run the `pipeline.sh` Script

- **Note:** The original pipeline process from the paper is in `pipeline.sh`. Feel free to run that script, but it won't generate a comprehensive log. The steps are the exact same as above except you use the following command when running the program:

  ```bash
  bash pipeline.sh > bash.log
  ```

## Re-run the Program

If you want to rerun the DeepNN program, you need to remove specific directories to reset the simulated data that you generated. Use the following commands:

```bash
rm -rf ~/dnn-paper/final_000
rm -rf ~/dnn-paper/final_001
rm -rf ~/dnn-paper/final_002
rm -rf ~/dnn-paper/paper
```