from pkg_resources import resource_filename
from lxml import etree
import os

from typing import Optional, Union

from ..gates._gml_gates import GMLRectangleGate, GMLBooleanGate, GMLPolygonGate, GMLQuadrantGate
from ..gates._gates import PolygonGate
from ..gates._wsp_gates import WSPEllipsoidGate

from ..gates._gate_utils import find_attribute_value

from ..transforms._matrix import Matrix
from ..transforms import _transforms, _wsp_transforms

import numpy as np



wsp_gate_constructor_lut = {
            'RectangleGate': GMLRectangleGate,
            'PolygonGate': GMLPolygonGate,
            'EllipsoidGate': WSPEllipsoidGate,
            'QuadrantGate': GMLQuadrantGate,
            'BooleanGate': GMLBooleanGate
        }

class FlowJoWorkspace:

    def __init__(self,
                 input_directory: str,
                 file_name: str) -> None:
        
        #self.resource_path = resource_filename("FACSPy", "_resources")
        self.original_filename = file_name
        
        self.ignore_transforms = False

        self.wsp_dict = self.parse_workspace(input_directory, file_name)

        

    #TODO: add typehinting dict
    def parse_workspace(self,
                        input_directory: str,
                        file_name: str) -> dict:
        (wsp_root,
         gating_namespace,
         data_type_namespace,
         transform_namespace) = self._extract_namespaces(input_directory, file_name)

        ns_map = wsp_root.nsmap
        
        sample_list: list[etree._Element] = self.parse_xml_samples(wsp_root, ns_map)
        
        group_node_list: list[etree._Element] = self.parse_xml_group_nodes(wsp_root, ns_map)

        raw_wsp_groups = self.parse_wsp_groups(group_node_list,
                                               ns_map,
                                               gating_namespace,
                                               data_type_namespace)
        
        raw_wsp_samples = self.parse_wsp_samples(sample_list, ns_map,
                                                 gating_namespace,
                                                 transform_namespace,
                                                 data_type_namespace)
    

    def parse_wsp_samples(self,
                          sample_list: list[etree._Element],
                          ns_map: dict,
                          gating_namespace: str,
                          transform_namespace: str,
                          data_type_namespace: str) -> dict:
        wsp_samples = {}

        for sample in sample_list:
            transforms: etree._Element = sample.find("Transformations", ns_map)
            keywords: etree._Element = sample.find("Keywords", ns_map)
            sample_node: etree._Element = sample.find("SampleNode", ns_map)
            
            sample_name = sample_node.attrib["name"]
            sample_id = sample_node.attrib["sampleID"]

            sample_xform_lut = self.parse_wsp_transforms(transforms, transform_namespace, data_type_namespace)

            sample_keywords_lut = self.parse_wsp_keywords(keywords)

            sample_comp = self.parse_wsp_compensation(sample, transform_namespace, data_type_namespace)

            sample_root_subpopulation = sample_node.find("Subpopulations", ns_map)
            if sample_root_subpopulation is None:
                sample_gates = []
            else:
                sample_gates = self.parse_wsp_subpopulations(
                    sample_root_subpopulation,
                    None,
                    gating_namespace,
                    data_type_namespace
                )

            # sample gate LUT will store everything we need to convert sample gates,
            # including any custom gates (ones with empty string owning groups).
            wsp_samples[sample_id] = {
                'sample_name': sample_name,
                'sample_gates': sample_gates,
                'custom_gate_ids': set(),
                'custom_gates': [],
                'transforms': sample_xform_lut,
                'keywords': sample_keywords_lut,
                'comp': sample_comp
            }

            for sample_gate in sample_gates:
                if sample_gate['owning_group'] == '':
                    # If the owning group is an empty string, it is a custom gate for that sample
                    # that is potentially used in another group. However, it appears that if a
                    # sample has a custom gate then that custom gate cannot be further customized.
                    # Since there is only a single custom gate per gate name per sample, then we
                    # can create a LUT of custom gates per sample
                    wsp_samples[sample_id]['custom_gate_ids'].add(sample_gate['gate'].gate_name)
                    wsp_samples[sample_id]['custom_gates'].append(
                        {
                            'gate': sample_gate['gate'],
                            'gate_path': sample_gate['gate_path']
                        }
                    )

        return wsp_samples         



    def parse_detectors(self,
                        matrix_element: etree._Element,
                        data_type_ns: str) -> list[str]:
        params_els: etree._Element = matrix_element.find(
            f'{data_type_ns}:parameters', matrix_element.nsmap
        )
        param_els: list[etree._Element] = params_els.findall(
            f'{data_type_ns}:parameter', matrix_element.nsmap
        )
        return [find_attribute_value(param_el, data_type_ns, 'name') for param_el in param_els]


    def parse_matrix(self,
                     matrix_element: etree._Element,
                     transform_ns: str) -> np.ndarray:
        
        return self.assemble_matrix(matrix_element)
    
    def parse_matrix_row(self,
                         coefficients: list[etree._Element]) -> np.ndarray:
        return np.array([find_attribute_value(float(coefficient)) for coefficient in coefficients]) 

    def parse_coefficients(self,
                           spill_element: etree._Element,
                           transform_ns: str) -> list[etree._Element]:
        return spill_element.findall(f'{transform_ns}:coefficient', spill_element.nsmap)

    def assemble_matrix(self,
                        matrix_element: etree._Element,
                        transform_ns: str):
        spill_els: list[etree._Element] = matrix_element.findall(
            f'{transform_ns}:spillover', matrix_element.nsmap
        )
        return np.ndarray([self.parse_matrix_row(self.parse_coefficients(spill_el, transform_ns)) for spill_el in spill_els])

    def parse_wsp_compensation(self,
                               sample: etree._Element,
                               transform_ns: str,
                               data_type_ns: str) -> Optional[dict]:
        
        matrix_elements = sample.findall(f'{transform_ns}:spilloverMatrix', sample.nsmap)

        if len(matrix_elements) > 1:
            raise ValueError("Multiple spillover matrices per sample are not supported.")
        if not matrix_elements:
            return None

        matrix_element: etree._Element = matrix_elements[0]

        detectors = self.parse_detectors(matrix_element, data_type_ns)
        matrix_array = self.parse_matrix(matrix_element, transform_ns)

        matrix = Matrix(
            matrix_id = matrix_element.attrib['name'],
            spill_data_or_file = matrix_array,
            detectors=detectors,
            fluorochromes=['' for _ in detectors]
        )

        return {
            'matrix_name': matrix_element.attrib['name'],
            'prefix': matrix_element.attrib['prefix'],
            'suffix': matrix_element.attrib['suffix'],
            'detectors': detectors,
            'matrix_array': matrix_array,
            'matrix': matrix,
        }


    def parse_wsp_keywords(self,
                           keywords: etree._Element) -> dict:
        
        keyword_els = keywords.getchildren()

        return {
            keyword_el.attrib['name']: keyword_el.attrib['value']
            for keyword_el in keyword_els
        }

    def parse_wsp_transforms(self,
                             transforms_el: etree._Element,
                             transform_ns: str,
                              data_type_ns: str) -> dict:
        
        xform_els: list[etree._Element] = transforms_el.getchildren()

        # there should be one transform per channel, use the channel names to create a LUT
        xforms_lut = {}

        for xform_el in xform_els:
            xform_type = xform_el.tag.partition('}')[-1]

            param_el = xform_el.find(f'{data_type_ns}:parameter', xform_el.nsmap)

            param_name = find_attribute_value(param_el, data_type_ns, 'name')

            # FlowKit only supports linear, log, and logicle transformations in FlowJo WSP files.
            # All other bi-ex transforms implemented by FlowJo are undocumented and not reproducible
            if xform_type == 'linear':
                min_range = find_attribute_value(xform_el, transform_ns, 'minRange')
                max_range = find_attribute_value(xform_el, transform_ns, 'maxRange')
                xforms_lut[param_name] = _transforms.LinearTransform(
                    param_name,
                    param_t=float(max_range),
                    param_a=float(min_range)
                )
            elif xform_type == 'log':
                offset = find_attribute_value(xform_el, transform_ns, 'offset')
                decades = find_attribute_value(xform_el, transform_ns, 'decades')
                xforms_lut[param_name] = _wsp_transforms.WSPLogTransform(
                    param_name,
                    offset=float(offset),
                    decades=float(decades)
                )
            elif xform_type == 'logicle':
                # logicle transform has 4 parameters: T, W, M, and A
                # these are attributes of the 'logicle' element
                param_t = find_attribute_value(xform_el, transform_ns, 'T')
                param_w = find_attribute_value(xform_el, transform_ns, 'W')
                param_m = find_attribute_value(xform_el, transform_ns, 'M')
                param_a = find_attribute_value(xform_el, transform_ns, 'A')
                xforms_lut[param_name] = _transforms.LogicleTransform(
                    param_name,
                    param_t=float(param_t),
                    param_w=float(param_w),
                    param_m=float(param_m),
                    param_a=float(param_a)
                )
            elif xform_type == 'biex':
                # biex transform has 5 parameters, but only 2 are really used
                # these are attributes of the 'biex' element
                param_neg = find_attribute_value(xform_el, transform_ns, 'neg')
                param_width = find_attribute_value(xform_el, transform_ns, 'width')
                param_length = find_attribute_value(xform_el, transform_ns, 'length')
                param_max_range = find_attribute_value(xform_el, transform_ns, 'maxRange')
                param_pos = find_attribute_value(xform_el, transform_ns, 'pos')
                param_pos = round(float(param_pos), 2)

                if param_length != '256':
                    raise ValueError(
                        f"FlowJo biex 'length' parameter value of {param_length} is not supported."
                    )

                xforms_lut[param_name] = _wsp_transforms.WSPBiexTransform(
                    param_name,
                    negative=float(param_neg),
                    width=float(param_width),
                    positive=float(param_pos),
                    max_value=float(param_max_range)
                )
            elif xform_type == 'fasinh':
                # FlowJo's implementation of fasinh is slightly different from GML,
                # and uses an additional 'length' scale factor. However, this scaling
                # doesn't seem to affect the results, and we can use the regular
                # GML version of asinh. The xform_el also contains other
                # unnecessary parameters: 'length', 'maxRange', and 'W'
                param_t = find_attribute_value(xform_el, transform_ns, 'T')
                param_a = find_attribute_value(xform_el, transform_ns, 'A')
                param_m = find_attribute_value(xform_el, transform_ns, 'M')
                xforms_lut[param_name] = _transforms.AsinhTransform(
                    param_name,
                    param_t=float(param_t),
                    param_m=float(param_m),
                    param_a=float(param_a)
                )
            else:
                error_msg = f"FlowJo transform type '{xform_type}' is undocumented and not supported in FlowKit. "
                error_msg += "Please edit the workspace in FlowJo and save all channel transformations as either " \
                                "linear, log, biex, logicle, or ArcSinh"

                raise ValueError(error_msg)

        return xforms_lut

    def parse_wsp_groups(self,
                         group_node_list: list[etree._Element],
                         ns_map: dict,
                         gating_namespace: str,
                         data_type_namespace: str) -> dict:
        wsp_groups = {}
        for group_node in group_node_list:
            group_name: str = group_node.attrib["name"]

            group_node_group: etree._Element = group_node.find("Group", ns_map)
            if group_node_group is not None:
                sample_ids = self.parse_sampleIDs_from_sample_refs(group_node_group, ns_map)
            
            group_node_subpopulation = group_node.find("Subpopulations", ns_map)
            if group_node_subpopulation is not None:
                group_gates = self.parse_wsp_subpopulations(
                    group_node_subpopulation,
                    None,
                    gating_namespace,
                    data_type_namespace
                )

            wsp_groups[group_name] = {
                "gates": group_gates,
                "samples": sample_ids
            }

        return wsp_groups

    def parse_wsp_subpopulations(self,
                                 subpopulation: etree._Element,
                                 gate_path: Optional[list],
                                 gating_ns: str,
                                 data_type_ns: str) -> list[dict]:
        
        ns_map = subpopulation.nsmap

        gates = []
        gate_path, parent_id = self.fetch_gate_path_and_parent(gate_path)
        
        populations: list[etree._Element] = subpopulation.findall("Population", ns_map)
        
        for population in populations:
            gates.append(self.fetch_gate_from_xml_population(population,
                                                             ns_map,
                                                             gating_ns,
                                                             data_type_ns,
                                                             parent_id,
                                                             gate_path))

            subpopulations = population.findall("Subpopulations", ns_map)
            child_gate_path = gate_path
            child_gate_path.append(population)
            for el in subpopulations:
                gates.extend(self.parse_wsp_subpopulations(el,
                                                           child_gate_path,
                                                           gating_ns,
                                                           data_type_ns))

        return gates

    def fetch_gate_from_xml_population(self,
                                       population: etree._Element,
                                       ns_map: dict,
                                       gating_ns: str,
                                       data_type_ns: str,
                                       parent_id: str,
                                       gate_path: list) -> dict:
        population_name = population.attrib["name"]
        owning_group = population.attrib["owningGroup"]
        
        gate_element = population.find("Gate", ns_map)
        gate_children: list[etree._Element] = gate_element.getchildren()

        gate_child_element = gate_children[0]
        # <Element {http://www.isac-net.org/std/Gating-ML/v2.0/gating}PolygonGate at 0x2206f425340>
        gate_type = gate_child_element.tag.partition("}")[-1]

        gate_class = wsp_gate_constructor_lut[gate_type]
        g: Union[GMLRectangleGate,
                 GMLPolygonGate,
                 WSPEllipsoidGate,
                 GMLQuadrantGate,
                 GMLBooleanGate] = gate_class(
                                       gate_child_element,
                                       gating_ns,
                                       data_type_ns
                                   )
        g.gate_name = population_name
        g.parent = parent_id

        return {'owning_group': owning_group,
                'gate': g,
                'gate_path': tuple(gate_path)}

    def fetch_gate_path_and_parent(self,
                                   gate_path: Optional[list]) -> tuple[list, str]:
        if gate_path is None:
            gate_path = ["root"]
            parent_id = None
        else:
            parent_id = gate_path[-1]

        return gate_path, parent_id

    def parse_sampleIDs_from_sample_refs(self,
                                         group_element: etree._Element,
                                         ns_map: dict) -> list[str]:
        sample_references: etree._Element = group_element.find("SampleRefs", ns_map)
        
        if sample_references is None:
            return []
        
        sample_reference_elements: list[etree._Element] = sample_references.findall("SampleRef", ns_map)
        return [sample_ref.attrib["sampleID"] for sample_ref in sample_reference_elements]
                  
        

    def parse_xml_group_nodes(self,
                              wsp_root: etree._Element,
                              ns_map: dict) -> list[etree._Element]:
        group_elements = self.parse_xml_groups(wsp_root, ns_map)
        return group_elements.findall("GroupNode", ns_map)

    def parse_xml_groups(self,
                         wsp_root: etree._Element,
                         ns_map: dict) -> etree._Element:
        return wsp_root.find("Groups", ns_map)

    def parse_xml_samples(self,
                          wsp_root: etree._Element,
                          ns_map: dict) -> list[etree._Element]:
        
        sample_list_elements = self.parse_xml_sample_list(wsp_root, ns_map)
        return sample_list_elements.findall("Sample", ns_map)

    def parse_xml_sample_list(self,
                              wsp_root: etree._Element,
                              ns_map: dict) -> etree._Element:
        return wsp_root.find("SampleList", ns_map)

    def _extract_namespaces(self,
                            input_directory,
                            file_name: str) -> tuple[etree._Element, str, str, str]:
        raw_wsp = etree.parse(os.path.join(input_directory, file_name))
        wsp_root = raw_wsp.getroot()
        gating_ns = None
        data_type_ns = None
        transform_ns = None

        # find GatingML target namespace in the map
        for ns, url in wsp_root.nsmap.items():
            if url == 'http://www.isac-net.org/std/Gating-ML/v2.0/gating':
                gating_ns = ns
            elif url == 'http://www.isac-net.org/std/Gating-ML/v2.0/datatypes':
                data_type_ns = ns
            elif url == 'http://www.isac-net.org/std/Gating-ML/v2.0/transformations':
                transform_ns = ns

        return wsp_root, gating_ns, data_type_ns, transform_ns



class DivaWorkspace:

    pass