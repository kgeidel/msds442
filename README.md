# MSDS 442: AI Agent Design and Development

<div align=center>
Kevin Geidel <br>
MSDS 442: AI Agent Design and Development<br>
Northwestern University<br>
Spring '25
</div>
<hr>

#### Assignment 1: Financial Planning Agent

To replicate the experiment run the following commands (note: these are form Linux/Unix based systems. Please adapt these commands for use with Windows if needed.)

```shell
# Clone and enter the repo
git clone git@github.com:kgeidel/msds442.git && cd msds442

# Create and activate the virtualenv for the project 
# (I am using pyenv, adapt for whatever Python environment manager you prefer.)
pyenv install 3.11.9
pyenv virtualenv 3.11.9 msds442
pyenv local msds442

# With the environment activated for this directory, install the required packages
pip install -r requirements.txt
```