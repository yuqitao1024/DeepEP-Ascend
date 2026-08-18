from dataclasses import dataclass


@dataclass(frozen=True)
class EPModeCase:
    do_handle_copy: bool
    expert_alignment: int
    use_fp8_dispatch: bool
    num_bias: int
    with_previous_event: bool
    async_with_compute_stream: bool
    allocate_on_comm_stream: bool

    @property
    def dtype_name(self) -> str:
        return "fp8" if self.use_fp8_dispatch else "bf16"

    @property
    def case_id(self) -> str:
        return (
            f"ep-{self.dtype_name}-align{self.expert_alignment}"
            f"-bias{self.num_bias}-hcopy{int(self.do_handle_copy)}"
            f"-prev{int(self.with_previous_event)}"
            f"-async{int(self.async_with_compute_stream)}"
            f"-alloc{int(self.allocate_on_comm_stream)}"
        )


def enumerate_ep_mode_cases() -> tuple[EPModeCase, ...]:
    cases = []
    for do_handle_copy in (True, False):
        for expert_alignment in (128, 1):
            for use_fp8_dispatch in (True, False):
                for num_bias in (0, 1, 2):
                    for with_previous_event in (False, True):
                        for async_with_compute_stream in (False, True):
                            allocations = (
                                (True,)
                                if with_previous_event
                                else (False, True)
                            )
                            for allocate_on_comm_stream in allocations:
                                cases.append(
                                    EPModeCase(
                                        do_handle_copy=do_handle_copy,
                                        expert_alignment=expert_alignment,
                                        use_fp8_dispatch=use_fp8_dispatch,
                                        num_bias=num_bias,
                                        with_previous_event=with_previous_event,
                                        async_with_compute_stream=(
                                            async_with_compute_stream
                                        ),
                                        allocate_on_comm_stream=(
                                            allocate_on_comm_stream
                                        ),
                                    )
                                )
    return tuple(cases)
