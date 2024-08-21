import os

def beep():
    # os.system("echo -n '\a';sleep 0.2;" * x)
    os.system("beep -f 1000q -l 1500")


# set up a mock system call that logs arguments to a list
call_log = []


def mock_system_call(cmd):
    call_log.append(cmd)


os.system = mock_system_call

# call the buzzer function
beep()

# verify that the system call was made with the expected arguments
assert len(call_log) == 1
assert call_log[0] == "beep -f 1000q -l 1500"
