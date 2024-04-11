from .dimension import QuadrantDivider, Dimension, RatioDimension


def find_attribute_value(xml_el, namespace, attribute_name) -> str:
    """
    Extract the value from an XML element attribute.
    :param xml_el: lxml etree Element
    :param namespace: string for the XML element's namespace prefix
    :param attribute_name: attribute string to retrieve the value from
    :return: value string for the given attribute_name
    """
    attribs = xml_el.xpath(
        f'@{namespace}:{attribute_name}',
        namespaces=xml_el.nsmap,
        smart_strings=False,
    )
    attribs_cnt = len(attribs)

    if attribs_cnt > 1:
        raise ValueError(
            "Multiple %s attributes found (line %d)" % (
                attribute_name, xml_el.sourceline
            )
        )
    elif attribs_cnt == 0:
        return None

    # return as pure str to save memory (otherwise it's an _ElementUnicodeResult from lxml)
    return str(attribs[0])


def _parse_dimension_element(
        dim_element,
        gating_namespace,
        data_type_namespace
):
    compensation_ref = find_attribute_value(dim_element, gating_namespace, 'compensation-ref')
    transformation_ref = find_attribute_value(dim_element, gating_namespace, 'transformation-ref')

    # should be 0 or only 1 'min' attribute (same for 'max')
    _min = find_attribute_value(dim_element, gating_namespace, 'min')
    _max = find_attribute_value(dim_element, gating_namespace, 'max')

    range_min = float(_min) if _min is not None else None
    range_max = float(_max) if _max is not None else None
    # ID be here
    fcs_dim_el = dim_element.find(
        f'{data_type_namespace}:fcs-dimension', namespaces=dim_element.nsmap
    )

    # if no 'fcs-dimension' element is present, this might be a
    # 'new-dimension'  made from a transformation on other dims
    if fcs_dim_el is None:
        new_dim_el = dim_element.find(
            f'{data_type_namespace}:new-dimension',
            namespaces=dim_element.nsmap,
        )
        if new_dim_el is None:
            raise ValueError(
                "Dimension invalid: neither fcs-dimension or new-dimension "
                "tags found (line %d)" % dim_element.sourceline
            )

        # if we get here, there should be a 'transformation-ref' attribute
        ratio_xform_ref = find_attribute_value(new_dim_el, data_type_namespace, 'transformation-ref')

        if ratio_xform_ref is None:
            raise ValueError(
                "New dimensions must provide a transform reference (line %d)" % dim_element.sourceline
            )
        return RatioDimension(
            ratio_xform_ref,
            compensation_ref,
            transformation_ref,
            range_min=range_min,
            range_max=range_max,
        )
    else:
        dim_id = find_attribute_value(fcs_dim_el, data_type_namespace, 'name')
        if dim_id is None:
            raise ValueError(
                'Dimension name not found (line %d)' % fcs_dim_el.sourceline
            )

        return Dimension(
            dim_id,
            compensation_ref,
            transformation_ref,
            range_min=range_min,
            range_max=range_max,
        )


def _parse_divider_element(divider_element, gating_namespace, data_type_namespace):
    # Get 'id' (present in quad gate dividers)
    divider_id = find_attribute_value(divider_element, gating_namespace, 'id')

    compensation_ref = find_attribute_value(divider_element, gating_namespace, 'compensation-ref')
    transformation_ref = find_attribute_value(divider_element, gating_namespace, 'transformation-ref')

    # ID be here
    fcs_dim_el = divider_element.find(
        f'{data_type_namespace}:fcs-dimension',
        namespaces=divider_element.nsmap,
    )

    dim_id = find_attribute_value(fcs_dim_el, data_type_namespace, 'name')
    if dim_id is None:
        raise ValueError(
            'Divider dimension name not found (line %d)' % fcs_dim_el.sourceline
        )

    # values in gating namespace, ok if not present
    value_els = divider_element.findall(
        f'{gating_namespace}:value', namespaces=divider_element.nsmap
    )

    values = [float(value.text) for value in value_els]
    return QuadrantDivider(
        divider_id, dim_id, compensation_ref, values, transformation_ref
    )


def parse_vertex_element(vertex_element, gating_namespace, data_type_namespace):
    """
    This class parses a GatingML-2.0 compatible vertex XML element and returns a list of coordinates.
    :param vertex_element: vertex XML element from a GatingML-2.0 document
    :param gating_namespace: XML namespace for gating elements/attributes
    :param data_type_namespace: XML namespace for data type elements/attributes
    """
    coordinates = []

    coord_els = vertex_element.findall(
        f'{gating_namespace}:coordinate', namespaces=vertex_element.nsmap
    )

    if len(coord_els) != 2:
        raise ValueError(
            'Vertex must contain 2 coordinate values (line %d)' % vertex_element.sourceline
        )

    # should be 0 or only 1 'min' attribute,
    for coord_el in coord_els:
        value = find_attribute_value(coord_el, data_type_namespace, 'value')
        if value is None:
            raise ValueError(
                'Vertex coordinate must have only 1 value (line %d)' % coord_el.sourceline
            )

        coordinates.append(float(value))

    return coordinates



def parse_gate_element(
        gate_element,
        gating_namespace,
        data_type_namespace
    ):
    """
    This class parses a GatingML-2.0 compatible gate XML element and extracts the gate ID,
     parent gate ID, and dimensions.
    :param gate_element: gate XML element from a GatingML-2.0 document
    :param gating_namespace: XML namespace for gating elements/attributes
    :param data_type_namespace: XML namespace for data type elements/attributes
    """
    gate_id = find_attribute_value(gate_element, gating_namespace, 'id')
    parent_id = find_attribute_value(gate_element, gating_namespace, 'parent_id')

    # most gates specify dimensions in the 'dimension' tag,
    # but quad gates specify dimensions in the 'divider' tag
    div_els = gate_element.findall(
        f'{gating_namespace}:divider', namespaces=gate_element.nsmap
    )

    dimensions = []  # may actually be a list of dividers

    if len(div_els) == 0:
        dim_els = gate_element.findall(
            f'{gating_namespace}:dimension', namespaces=gate_element.nsmap
        )

        dimensions = []

        for dim_el in dim_els:
            dim = _parse_dimension_element(dim_el, gating_namespace, data_type_namespace)
            dimensions.append(dim)
    else:
        for div_el in div_els:
            dim = _parse_divider_element(div_el, gating_namespace, data_type_namespace)
            dimensions.append(dim)

    return gate_id, parent_id, dimensions
