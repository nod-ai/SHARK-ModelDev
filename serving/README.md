# Turbine Serving Infrastructure

This sub-project contains components and infrastructure for serving various
forms of Turbine compiled models. Instead of coming with models, it defines
ABIs that compiled models should adhere to in order to be served. It then
allows them to be delivered as web endpoints via popular APIs.

As emulation can be the sincerest form of flattery, this project derives
substantial inspiration from vllm and the OpenAI APIs, emulating and
interopping with them where possible. It is intended to be the lightest
weight possible reference implementation for serving models with an
opinionated compiled form, built elsewhere in the project.
