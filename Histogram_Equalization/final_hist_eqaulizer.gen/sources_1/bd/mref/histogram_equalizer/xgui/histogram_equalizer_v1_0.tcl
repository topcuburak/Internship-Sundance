# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "height" -parent ${Page_0}
  ipgui::add_param $IPINST -name "max_pixel_val" -parent ${Page_0}
  ipgui::add_param $IPINST -name "width" -parent ${Page_0}


}

proc update_PARAM_VALUE.height { PARAM_VALUE.height } {
	# Procedure called to update height when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.height { PARAM_VALUE.height } {
	# Procedure called to validate height
	return true
}

proc update_PARAM_VALUE.max_pixel_val { PARAM_VALUE.max_pixel_val } {
	# Procedure called to update max_pixel_val when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.max_pixel_val { PARAM_VALUE.max_pixel_val } {
	# Procedure called to validate max_pixel_val
	return true
}

proc update_PARAM_VALUE.width { PARAM_VALUE.width } {
	# Procedure called to update width when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.width { PARAM_VALUE.width } {
	# Procedure called to validate width
	return true
}


proc update_MODELPARAM_VALUE.width { MODELPARAM_VALUE.width PARAM_VALUE.width } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.width}] ${MODELPARAM_VALUE.width}
}

proc update_MODELPARAM_VALUE.height { MODELPARAM_VALUE.height PARAM_VALUE.height } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.height}] ${MODELPARAM_VALUE.height}
}

proc update_MODELPARAM_VALUE.max_pixel_val { MODELPARAM_VALUE.max_pixel_val PARAM_VALUE.max_pixel_val } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.max_pixel_val}] ${MODELPARAM_VALUE.max_pixel_val}
}

