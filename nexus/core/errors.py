"""Nexus 异常定义"""


class NexusError(Exception):
    pass


class ProviderError(NexusError):
    """模型调用失败"""
    pass


class ProviderNotFoundError(NexusError):
    """未找到指定的 provider"""
    pass


class ConstraintError(NexusError):
    """约束验证失败"""
    pass
