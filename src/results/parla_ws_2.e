
Currently Loaded Modules:
  1) intel/19.1.1   4) autotools/1.2   7) pmix/3.1.4     10) TACC
  2) impi/19.0.9    5) python3/3.7.0   8) hwloc/1.11.12  11) cuda/11.0 (g)
  3) git/2.24.1     6) cmake/3.20.3    9) xalt/2.10.13

  Where:
   g:  built for GPU

 

Exception in task
Traceback (most recent call last):
  File "/home1/03727/tg830270/Research/Parla.py/parla/task_runtime.py", line 283, in run
    task_state = self._state.func(self, *self._state.args)
  File "/home1/03727/tg830270/Research/Parla.py/parla/tasks.py", line 300, in _task_callback
    new_task_info = body.send(in_value)
  File "tsqr_parla_mpi.py", line 546, in launch_tsqr
    Qr, Rr = await tsqr_blocked_mpi(Ar,comm, BLOCK_SIZE)
  File "tsqr_parla_mpi.py", line 335, in tsqr_blocked_mpi
    assert ncols <= block_size, "Block size must be greater than or equal to the number of columns in the input matrix"
AssertionError: Block size must be greater than or equal to the number of columns in the input matrix
Traceback (most recent call last):
  File "tsqr_parla_mpi.py", line 696, in <module>
    main()
  File "/home1/03727/tg830270/Research/Parla.py/parla/__init__.py", line 34, in __exit__
    return self._sched.__exit__(exc_type, exc_val, exc_tb)
  File "/home1/03727/tg830270/Research/Parla.py/parla/task_runtime.py", line 679, in __exit__
    raise self._exceptions[0]
  File "/home1/03727/tg830270/Research/Parla.py/parla/task_runtime.py", line 283, in run
    task_state = self._state.func(self, *self._state.args)
  File "/home1/03727/tg830270/Research/Parla.py/parla/tasks.py", line 300, in _task_callback
    new_task_info = body.send(in_value)
  File "tsqr_parla_mpi.py", line 546, in launch_tsqr
    Qr, Rr = await tsqr_blocked_mpi(Ar,comm, BLOCK_SIZE)
  File "tsqr_parla_mpi.py", line 335, in tsqr_blocked_mpi
    assert ncols <= block_size, "Block size must be greater than or equal to the number of columns in the input matrix"
AssertionError: Block size must be greater than or equal to the number of columns in the input matrix
Exception in task
Traceback (most recent call last):
  File "/home1/03727/tg830270/Research/Parla.py/parla/task_runtime.py", line 283, in run
    task_state = self._state.func(self, *self._state.args)
  File "/home1/03727/tg830270/Research/Parla.py/parla/tasks.py", line 300, in _task_callback
    new_task_info = body.send(in_value)
  File "tsqr_parla_mpi.py", line 546, in launch_tsqr
    Qr, Rr = await tsqr_blocked_mpi(Ar,comm, BLOCK_SIZE)
  File "tsqr_parla_mpi.py", line 335, in tsqr_blocked_mpi
    assert ncols <= block_size, "Block size must be greater than or equal to the number of columns in the input matrix"
AssertionError: Block size must be greater than or equal to the number of columns in the input matrix
Traceback (most recent call last):
  File "tsqr_parla_mpi.py", line 696, in <module>
    main()
  File "/home1/03727/tg830270/Research/Parla.py/parla/__init__.py", line 34, in __exit__
    return self._sched.__exit__(exc_type, exc_val, exc_tb)
  File "/home1/03727/tg830270/Research/Parla.py/parla/task_runtime.py", line 679, in __exit__
    raise self._exceptions[0]
  File "/home1/03727/tg830270/Research/Parla.py/parla/task_runtime.py", line 283, in run
    task_state = self._state.func(self, *self._state.args)
  File "/home1/03727/tg830270/Research/Parla.py/parla/tasks.py", line 300, in _task_callback
    new_task_info = body.send(in_value)
  File "tsqr_parla_mpi.py", line 546, in launch_tsqr
    Qr, Rr = await tsqr_blocked_mpi(Ar,comm, BLOCK_SIZE)
  File "tsqr_parla_mpi.py", line 335, in tsqr_blocked_mpi
    assert ncols <= block_size, "Block size must be greater than or equal to the number of columns in the input matrix"
AssertionError: Block size must be greater than or equal to the number of columns in the input matrix
