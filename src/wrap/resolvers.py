import re

from omegaconf.omegaconf import OmegaConf, Resolver


def register_resolver(name: str, **kwargs):
    def register(resolver: Resolver):
        OmegaConf.register_new_resolver(name=name, resolver=resolver, **kwargs)
        return resolver
    return register


@register_resolver("last")
def last(s: str, delim: str):
    return s.rpartition(delim)[-1]


@register_resolver("safe")
def safe(s: str):
    return re.sub(re.compile('[^a-zA-Z0-9_]'), '-', s)
