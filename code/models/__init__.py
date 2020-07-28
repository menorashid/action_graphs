def get(name,args):
    import importlib
    if args is not None:
    	return importlib.import_module("models.%s" % name).Network(**args)
    else:
    	return importlib.import_module("models.%s" % name).Network()
