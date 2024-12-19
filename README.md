# DeepNN

DeepNN is a deep learning framework designed for Professor Xiao Liu's class. This guide provides step-by-step instructions to help you set up and run DeepNN on your local machine.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
  - [1. Install Anaconda](#1-install-anaconda)
  - [2. Create and Activate a Virtual Environment](#2-create-and-activate-a-virtual-environment)
  - [3. Install R Data.Table Package](#3-install-r-datatable-package)
- [Clone the Repository](#clone-the-repository)
- [Install Necessary Python Packages](#install-necessary-python-packages)
- [Run the Pipeline](#run-the-pipeline)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- [Anaconda](https://www.anaconda.com/products/distribution) (for managing environments)
- [Git](https://git-scm.com/downloads) (for cloning the repository)
- [Python 3.8](https://www.python.org/downloads/release/python-380/)
- [R](https://www.r-project.org/) (required for the `r-data.table` package)

## Setup Instructions

Follow these steps to set up DeepNN on your machine.

### 1. Install Anaconda

If you haven't installed Anaconda yet, download and install it from the [official website](https://www.anaconda.com/products/distribution). Follow the installation instructions for your operating system.

### 2. Create and Activate a Virtual Environment

Creating a virtual environment helps manage dependencies and avoid conflicts.

1. **Open your terminal or command prompt.**

2. **Create a new Conda environment named `deepnn` with Python 3.8:**

   ```bash
   conda create --name deepnn python=3.8
   ```

3. **Activate the `deepnn` environment:**

   ```bash
   conda activate deepnn
   ```

### 3. Install R Data.Table Package

DeepNN requires the `r-data.table` package from Conda-Forge.

1. **Install `r-data.table` using Conda:**

   ```bash
   conda install -c conda-forge r-data.table
   ```

   - **Explanation:** This command installs the `r-data.table` package from the Conda-Forge channel.

## Clone the Repository

Next, clone the DeepNN repository from GitHub to your local machine.

1. **Navigate to the directory where you want to clone the repository:**

   ```bash
   cd /path/to/your/preferred/directory
   ```

   - **Replace `/path/to/your/preferred/directory` with your desired path.**

2. **Clone the DeepNN repository:**

   ```bash
   git clone https://github.com/sichenz/DeepNN.git
   ```

   - **Explanation:** This command clones the DeepNN repository to your current directory.

3. **Navigate into the cloned repository:**

   ```bash
   cd DeepNN
   ```

## Install Necessary Python Packages

Install the required Python packages to ensure DeepNN runs smoothly.

1. **Upgrade `pip`, `setuptools`, and `wheel`:**

   ```bash
   pip install --upgrade pip setuptools wheel
   ```

   - **Explanation:** This command upgrades `pip`, `setuptools`, and `wheel` to their latest versions.

2. **Run the setup script to install additional dependencies:**

   ```bash
   bash setup/run_setup.sh
   ```

   - **Explanation:** Executes the `run_setup.sh` script located in the `setup` directory to install necessary packages.

   - **Note:** Ensure that the `run_setup.sh` script has execute permissions. If not, you can make it executable with:

     ```bash
     chmod +x setup/run_setup.sh
     ```

## Run the Pipeline

After setting up the environment and installing all dependencies, you can run the DeepNN pipeline.

1. **Open the `pipeline.sh` script in a text editor:**

   ```bash
   nano pipeline.sh
   ```

   - **Alternatively, use any text editor of your choice (e.g., VS Code, Sublime Text).**

2. **Locate the `cd` command in the script and update it to point to the absolute path of your cloned `DeepNN` repository.**

   - **Example:**

     ```bash
     cd /Users/yourusername/path/to/DeepNN
     ```

   - **Ensure that you replace `/Users/yourusername/path/to/DeepNN` with the actual path on your system.**

3. **Save and close the `pipeline.sh` script.**

4. **Make sure the `pipeline.sh` script has execute permissions:**

   ```bash
   chmod +x pipeline.sh
   ```

5. **Run the pipeline script:**

   ```bash
   bash pipeline.sh
   ```

   - **Explanation:** Executes the `pipeline.sh` script to start the DeepNN pipeline.

## Troubleshooting

- **Conda Activation Issues:**
  - If you encounter issues activating the Conda environment, ensure that Anaconda is correctly installed and that your terminal recognizes Conda commands. You might need to initialize Conda:

    ```bash
    conda init
    ```

    - **Then restart your terminal.**

- **Permission Denied Errors:**
  - If you receive permission errors when running scripts, ensure that the scripts have execute permissions:

    ```bash
    chmod +x script_name.sh
    ```

- **Missing Dependencies:**
  - If the setup script fails, verify that all prerequisites are installed and that you are within the correct Conda environment.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the repository.**
2. **Create a new branch for your feature or bugfix:**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit your changes:**

   ```bash
   git commit -m "Add YourFeature"
   ```

4. **Push to your fork:**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a pull request on GitHub.**

## License

This project is licensed under the [MIT License](LICENSE).

---

*For any further questions or support, please open an issue on the [GitHub repository](https://github.com/sichenz/DeepNN/issues).*