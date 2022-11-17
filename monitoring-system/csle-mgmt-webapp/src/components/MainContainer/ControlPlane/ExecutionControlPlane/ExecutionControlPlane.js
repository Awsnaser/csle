import React, {useState} from 'react';
import './ExecutionControlPlane.css';
import Card from 'react-bootstrap/Card';
import Button from 'react-bootstrap/Button'
import Table from 'react-bootstrap/Table'
import Accordion from 'react-bootstrap/Accordion';
import Collapse from 'react-bootstrap/Collapse'
import getIps from "../../../Common/getIps";
import getTopicsString from "../../../Common/getTopicsString";
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Tooltip from 'react-bootstrap/Tooltip';
import Spinner from 'react-bootstrap/Spinner'

/**
 * Component representing the /emulation-executions/<id>/control resource
 */
const ExecutionControlPlane = (props) => {
    const [runningContainersOpen, setRunningContainersOpen] = useState(false);
    const [activeNetworksOpen, setActiveNetworksOpen] = useState(false);
    const [clientManagersOpen, setClientManagersOpen] = useState(false);
    const [dockerStatsManagersOpen, setDockerStatsManagersOpen] = useState(false);
    const [hostManagersOpen, setHostManagersOpen] = useState(false);
    const [kafkaManagersOpen, setKafkaManagersOpen] = useState(false);
    const [ossecIdsManagersOpen, setOssecIdsManagersOpen] = useState(false);
    const [snortManagersOpen, setSnortManagersOpen] = useState(false);
    const [elkManagersOpen, setElkManagersOpen] = useState(false);
    const [trafficManagersOpen, setTrafficManagersOpen] = useState(false);
    const [loadingEntities, setLoadingEntities] = useState([]);
    console.log(props.info)

    const activeStatus = (active) => {
        if (active) {
            return (<td className="containerRunningStatus">Active</td>)
        } else {
            return (<td className="containerStoppedStatus">Inactive</td>)
        }
    }

    const renderStopTooltip = (props) => {
        return (<Tooltip id="button-tooltip" {...props} className="toolTipRefresh">
            Stop
        </Tooltip>)
    }

    const renderStartTooltip = (props) => {
        return (<Tooltip id="button-tooltip" {...props} className="toolTipRefresh">
            Start
        </Tooltip>)
    }

    const addLoadingEntity = (entity) => {
        var newLoadingEntities= []
        for (let i = 0; i < loadingEntities.length; i++) {
            newLoadingEntities.push(loadingEntities[i])
        }
        newLoadingEntities.push(entity)
        setLoadingEntities(newLoadingEntities)
    }

    const removeLoadingEntity = (entity) => {
        var newLoadingEntities= []
        for (let i = 0; i < loadingEntities.length; i++) {
            if(loadingEntities[i] !== entity) {
                newLoadingEntities.push(loadingEntities[i])
            }
        }
        setLoadingEntities(newLoadingEntities)
    }

    const startOrStop = (start, stop, entity) => {
        addLoadingEntity(entity)
        props.startOrStopEntity(props.execution.ip_first_octet, props.execution.emulation_name,
            start, stop)
        // props.startOrStopContainer(props.execution.ip_first_octet, props.execution.emulation_name, start, stop, entity)
    }

    const SpinnerOrButton = (props) => {
        if (props.loading) {
            return (<Spinner
                as="span"
                animation="grow"
                size="sm"
                role="status"
                aria-hidden="true"
            />)
        } else {
            if (props.running) {
                return (
                    <OverlayTrigger
                        placement="right"
                        delay={{show: 0, hide: 0}}
                        overlay={renderStopTooltip}
                    >
                        <Button variant="warning" className="startButton" size="sm"
                                onClick={() => startOrStop(false, true, props.entity, props.name, props.ip)}>
                            <i className="fa fa-stop-circle-o startStopIcon" aria-hidden="true"/>
                        </Button>
                    </OverlayTrigger>
                )
            } else {
                return (
                    <OverlayTrigger
                        placement="right"
                        delay={{show: 0, hide: 0}}
                        overlay={renderStartTooltip}
                    >
                        <Button variant="success" className="startButton" size="sm"
                                onClick={() => startOrStop(true, false, props.entity)}>
                            <i className="fa fa-play startStopIcon" aria-hidden="true"/>
                        </Button>
                    </OverlayTrigger>
                )
            }
        }
    };

    return (<Card key={props.execution.name} ref={props.wrapper}>
        <Card.Header>
            <Accordion.Toggle as={Button} variant="link" eventKey={props.execution.emulation_name + "_"
                + props.execution.ip_first_octet} className="mgHeader">
                <span className="subnetTitle">ID: {props.execution.ip_first_octet}, name: {props.execution.emulation_name}</span>
            </Accordion.Toggle>
        </Card.Header>
        <Accordion.Collapse eventKey={props.execution.emulation_name + "_" + props.execution.ip_first_octet}>
            <Card.Body>

                <Card className="subCard">
                    <Card.Header>
                        <Button
                            onClick={() => setRunningContainersOpen(!runningContainersOpen)}
                            aria-controls="runningContainersBody"
                            aria-expanded={runningContainersOpen}
                            variant="link"
                        >
                            <h5 className="semiTitle"> Containers statuses </h5>
                        </Button>
                    </Card.Header>
                    <Collapse in={runningContainersOpen}>
                        <div id="activeNetworksBody" className="cardBodyHidden">
                            <div className="table-responsive">
                                <Table striped bordered hover>
                                    <thead>
                                    <tr>
                                        <th>Name</th>
                                        <th>Image</th>
                                        <th>Os</th>
                                        <th>IPs</th>
                                        <th>Status</th>
                                        <th>Actions</th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    {props.info.running_containers.map((container, index) =>
                                        <tr key={container.full_name_str + "-" + index}>
                                            <td>{container.full_name_str}</td>
                                            <td>{container.name}</td>
                                            <td>{container.os}</td>
                                            <td>{getIps(container.ips_and_networks).join(", ")}</td>
                                            <td className="containerRunningStatus"> Running</td>
                                            <td>
                                                <SpinnerOrButton
                                                    loading={loadingEntities.includes(container.full_name_str)}
                                                    running={true} entity="container"
                                                    name={container.full_name_str} ip={container.ips_and_networks[0]}
                                                />
                                            </td>
                                        </tr>
                                    )}
                                    {props.info.stopped_containers.map((container, index) =>
                                        <tr key={container.full_name_str + "-" + index}>
                                            <td>{container.full_name_str}</td>
                                            <td>{container.name}</td>
                                            <td>{container.os}</td>
                                            <td>{getIps(container.ips_and_networks).join(", ")}</td>
                                            <td className="containerStoppedStatus">Stopped</td>
                                            <td>
                                                <SpinnerOrButton
                                                    loading={loadingEntities.includes(container.full_name_str)}
                                                    running={false} entity="container"
                                                    name={container.full_name_str} ip={container.ips_and_networks[0]} />
                                            </td>
                                        </tr>
                                    )}
                                    </tbody>
                                </Table>
                            </div>
                        </div>
                    </Collapse>
                </Card>

                <Card className="subCard">
                    <Card.Header>
                        <Button
                            onClick={() => setActiveNetworksOpen(!activeNetworksOpen)}
                            aria-controls="activeNetworksBody"
                            aria-expanded={activeNetworksOpen}
                            variant="link"
                        >
                            <h5 className="semiTitle"> Active networks </h5>
                        </Button>
                    </Card.Header>
                    <Collapse in={activeNetworksOpen}>
                        <div id="activeNetworksBody" className="cardBodyHidden">
                            <div className="table-responsive">
                                <Table striped bordered hover>
                                    <thead>
                                    <tr>
                                        <th>Network name</th>
                                        <th>Subnet mask</th>
                                        <th>Bitmask</th>
                                        <th>Status</th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    {props.info.active_networks.map((network, index) =>
                                        <tr key={network.name + "-" + index}>
                                            <td>{network.name}</td>
                                            <td>{network.subnet_mask}</td>
                                            <td>{network.bitmask}</td>
                                            <td className="containerRunningStatus">Active</td>
                                        </tr>
                                    )}
                                    {props.info.inactive_networks.map((network, index) =>
                                        <tr key={network.name + "-" + index}>
                                            <td>{network.name}</td>
                                            <td>{network.subnet_mask}</td>
                                            <td>{network.bitmask}</td>
                                            <td className="containerStoppedStatus">Inactive</td>
                                        </tr>
                                    )}
                                    </tbody>
                                </Table>
                            </div>
                        </div>
                    </Collapse>
                </Card>

                <Card className="subCard">
                    <Card.Header>
                        <Button
                            onClick={() => setClientManagersOpen(!clientManagersOpen)}
                            aria-controls="clientManagersBody"
                            aria-expanded={clientManagersOpen}
                            variant="link"
                        >
                            <h5 className="semiTitle"> Client managers</h5>
                        </Button>
                    </Card.Header>
                    <Collapse in={clientManagersOpen}>
                        <div id="clientManagersBody" className="cardBodyHidden">
                            <div className="table-responsive">
                                <Table striped bordered hover>
                                    <thead>
                                    <tr>
                                        <th>Service</th>
                                        <th>IP</th>
                                        <th>Port</th>
                                        <th>Status</th>
                                        <th># Clients</th>
                                        <th>Time-step length (s)</th>
                                        <th>Actions</th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    {props.info.client_managers_info.client_managers_statuses.map((status, index) =>
                                        <tr key={"client-manager-" + index}>
                                            <td>Client manager</td>
                                            <td>{props.info.client_managers_info.ips[index]}</td>
                                            <td>{props.info.client_managers_info.ports[index]}</td>
                                            {activeStatus(props.info.client_managers_info.running)}
                                            <td></td>
                                            <td>{status.clients_time_step_len_seconds}</td>
                                            <td>
                                                <SpinnerOrButton
                                                    loading={loadingEntities.includes("client-manager-"+
                                                        props.info.client_managers_info.ips[index])}
                                                    running={props.info.client_managers_info.running}
                                                    entity={"client-manager"} name={"client-manager"}
                                                    ip={props.info.client_managers_info.ips[index]}
                                                />
                                            </td>
                                        </tr>
                                    )}
                                    {props.info.client_managers_info.client_managers_statuses.map((status, index) =>
                                        <tr key={"client-population-" + index}>
                                            <td>Client process</td>
                                            <td>{props.info.client_managers_info.ips[index]}</td>
                                            <td></td>
                                            {activeStatus(status.client_process_active)}
                                            <td>{status.num_clients}</td>
                                            <td>{status.clients_time_step_len_seconds}</td>
                                            <td>
                                                <SpinnerOrButton
                                                    loading={loadingEntities.includes("client-population-"+
                                                        props.info.client_managers_info.ips[index])}
                                                    running={status.client_process_active}
                                                    entity={"client-population"} name={"client-population"}
                                                    ip={props.info.client_managers_info.ips[index]}
                                                />
                                            </td>
                                        </tr>
                                    )}
                                    {props.info.client_managers_info.client_managers_statuses.map((status, index) =>
                                        <tr key={"client-producer-" + index}>
                                            <td>Producer process</td>
                                            <td>{props.info.client_managers_info.ips[index]}</td>
                                            <td></td>
                                            {activeStatus(status.producer_active)}
                                            <td></td>
                                            <td>{status.clients_time_step_len_seconds}</td>
                                            <td>
                                                <SpinnerOrButton
                                                    loading={loadingEntities.includes("client-producer-"+
                                                        props.info.client_managers_info.ips[index])}
                                                    running={status.producer_active}
                                                    entity={"client-producer"} name={"client-producer"}
                                                    ip={props.info.client_managers_info.ips[index]}
                                                />
                                            </td>
                                        </tr>
                                    )}
                                    </tbody>
                                </Table>
                            </div>
                        </div>
                    </Collapse>
                </Card>

                <Card className="subCard">
                    <Card.Header>
                        <Button
                            onClick={() => setDockerStatsManagersOpen(!dockerStatsManagersOpen)}
                            aria-controls="dockerStatsManagersBody"
                            aria-expanded={dockerStatsManagersOpen}
                            variant="link"
                        >
                            <h5 className="semiTitle"> Docker Statistics Managers </h5>
                        </Button>
                    </Card.Header>
                    <Collapse in={dockerStatsManagersOpen}>
                        <div id="dockerStatsManagersBody" className="cardBodyHidden">
                            <div className="table-responsive">
                                <Table striped bordered hover>
                                    <thead>
                                    <tr>
                                        <th>Service</th>
                                        <th>IP</th>
                                        <th>Port</th>
                                        <th>Status</th>
                                        <th>Actions</th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    {props.info.docker_stats_managers_info.docker_stats_managers_statuses.map((status, index) =>
                                        <tr key={"docker_stats_status-" + index}>
                                            <td>Docker Statistics Manager</td>
                                            <td>{props.info.docker_stats_managers_info.ips[index]}</td>
                                            <td>{props.info.docker_stats_managers_info.ports[index]}</td>
                                            {activeStatus(props.info.docker_stats_managers_info.running)}
                                            <td>
                                            </td>
                                        </tr>
                                    )}
                                    {props.info.docker_stats_managers_info.docker_stats_managers_statuses.map((status, index) =>
                                        <tr key={"docker_stats_status-" + index}>
                                            <td>Docker Statistics Monitor Thread</td>
                                            <td>{props.info.docker_stats_managers_info.ips[index]}</td>
                                            <td></td>
                                            {activeStatus(status.num_monitors > 0)}
                                            <td>
                                                <SpinnerOrButton
                                                    loading={loadingEntities.includes("docker_stats_manager")}
                                                    running={status.num_monitors > 0}
                                                    entity={"docker_stats_manager"}/>
                                            </td>
                                        </tr>
                                    )}
                                    </tbody>
                                </Table>
                            </div>
                        </div>
                    </Collapse>
                </Card>

                <Card className="subCard">
                    <Card.Header>
                        <Button
                            onClick={() => setHostManagersOpen(!hostManagersOpen)}
                            aria-controls="hostManagersBody"
                            aria-expanded={hostManagersOpen}
                            variant="link"
                        >
                            <h5 className="semiTitle"> Host managers </h5>
                        </Button>
                    </Card.Header>
                    <Collapse in={hostManagersOpen}>
                        <div id="hostManagersOpen" className="cardBodyHidden">
                            <div className="table-responsive">
                                <Table striped bordered hover>
                                    <thead>
                                    <tr>
                                        <th>Service</th>
                                        <th>IP</th>
                                        <th>Port</th>
                                        <th>Status</th>
                                        <th>Actions</th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    {props.info.host_managers_info.host_managers_statuses.map((status, index) =>
                                        <tr key={"host_manager_status-" + index}>
                                            <td>Host Manager</td>
                                            <td>{props.info.host_managers_info.ips[index]}</td>
                                            <td>{props.info.host_managers_info.ports[index]}</td>
                                            {activeStatus(props.info.host_managers_info.running)}
                                            <td>
                                            </td>
                                        </tr>
                                    )}
                                    {props.info.host_managers_info.host_managers_statuses.map((status, index) =>
                                        <tr key={"host_manager_status-" + index}>
                                            <td>Host monitor thread</td>
                                            <td>{props.info.host_managers_info.ips[index]}</td>
                                            <td></td>
                                            {activeStatus(status.running)}
                                            <td>
                                                <SpinnerOrButton
                                                    loading={loadingEntities.includes("host_manager")}
                                                    running={status.running}
                                                    entity={"host_manager"}/>
                                            </td>
                                        </tr>
                                    )}
                                    </tbody>
                                </Table>
                            </div>
                        </div>
                    </Collapse>
                </Card>

                <Card className="subCard">
                    <Card.Header>
                        <Button
                            onClick={() => setKafkaManagersOpen(!kafkaManagersOpen)}
                            aria-controls="kafkaManagersBody"
                            aria-expanded={kafkaManagersOpen}
                            variant="link"
                        >
                            <h5 className="semiTitle"> Kafka managers </h5>
                        </Button>
                    </Card.Header>
                    <Collapse in={kafkaManagersOpen}>
                        <div id="kafkaManagersBody" className="cardBodyHidden">
                            <div className="table-responsive">
                                <Table striped bordered hover>
                                    <thead>
                                    <tr>
                                        <th>Service</th>
                                        <th>IP</th>
                                        <th>Port</th>
                                        <th>Topics</th>
                                        <th>Status</th>
                                        <th>Actions</th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    {props.info.kafka_managers_info.kafka_managers_statuses.map((status, index) =>
                                        <tr key={"kafka_manager_status-" + index}>
                                            <td>Kafka Manager</td>
                                            <td>{props.info.kafka_managers_info.ips[index]}</td>
                                            <td>{props.info.kafka_managers_info.ports[index]}</td>
                                            <td></td>
                                            {activeStatus(props.info.kafka_managers_info.running)}
                                            <td>
                                            </td>
                                        </tr>
                                    )}
                                    {props.info.kafka_managers_info.kafka_managers_statuses.map((status, index) =>
                                        <tr key={"kafka_manager_status-" + index}>
                                            <td>Kafka</td>
                                            <td>{props.info.kafka_managers_info.ips[index]}</td>
                                            <td>{props.execution.emulation_env_config.kafka_config.kafka_port}</td>
                                            <td>{getTopicsString(status.topics)}</td>
                                            {activeStatus(status.running)}
                                            <td>
                                                <SpinnerOrButton
                                                    loading={loadingEntities.includes("kafka_manager")}
                                                    running={status.running}
                                                    entity={"kafka_manager"}/>
                                            </td>
                                        </tr>
                                    )}
                                    </tbody>
                                </Table>
                            </div>
                        </div>
                    </Collapse>
                </Card>

                <Card className="subCard">
                    <Card.Header>
                        <Button
                            onClick={() => setOssecIdsManagersOpen(!ossecIdsManagersOpen)}
                            aria-controls="ossecIdsManagersBody"
                            aria-expanded={ossecIdsManagersOpen}
                            variant="link"
                        >
                            <h5 className="semiTitle"> OSSEC IDS Managers</h5>
                        </Button>
                    </Card.Header>
                    <Collapse in={ossecIdsManagersOpen}>
                        <div id="ossecIdsManagersBody" className="cardBodyHidden">
                            <div className="table-responsive">
                                <Table striped bordered hover>
                                    <thead>
                                    <tr>
                                        <th>Service</th>
                                        <th>IP</th>
                                        <th>Port</th>
                                        <th>Status</th>
                                        <th>Actions</th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    {props.info.ossec_managers_info.ossec_ids_statuses.map((status, index) =>
                                        <tr key={"ossec_manager_status-" + index}>
                                            <td>OSSEC IDS Manager</td>
                                            <td>{props.info.ossec_managers_info.ips[index]}</td>
                                            <td>{props.info.ossec_managers_info.ports[index]}</td>
                                            {activeStatus(props.info.ossec_managers_info.running)}
                                            <td>
                                            </td>
                                        </tr>
                                    )}
                                    {props.info.ossec_managers_info.ossec_ids_statuses.map((status, index) =>
                                        <tr key={"ossec_manager_status-" + index}>
                                            <td>OSSEC IDS</td>
                                            <td>{props.info.ossec_managers_info.ips[index]}</td>
                                            <td></td>
                                            {activeStatus(status.running)}
                                            <td>
                                                <SpinnerOrButton
                                                    loading={loadingEntities.includes("ossec_ids_manager")}
                                                    running={status.running}
                                                    entity={"ossec_ids_manager"}/>
                                            </td>
                                        </tr>
                                    )}
                                    </tbody>
                                </Table>
                            </div>
                        </div>
                    </Collapse>
                </Card>

                <Card className="subCard">
                    <Card.Header>
                        <Button
                            onClick={() => setSnortManagersOpen(!snortManagersOpen)}
                            aria-controls="snortManagersBody "
                            aria-expanded={snortManagersOpen}
                            variant="link"
                        >
                            <h5 className="semiTitle"> Snort IDS Managers</h5>
                        </Button>
                    </Card.Header>
                    <Collapse in={snortManagersOpen}>
                        <div id="snortManagersBody" className="cardBodyHidden">
                            <div className="table-responsive">
                                <Table striped bordered hover>
                                    <thead>
                                    <tr>
                                        <th>Service</th>
                                        <th>IP</th>
                                        <th>Port</th>
                                        <th>Status</th>
                                        <th>Actions</th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    {props.info.snort_managers_info.snort_statuses.map((status, index) =>
                                        <tr key={"snort_manager_status-" + index}>
                                            <td>Snort IDS Manager</td>
                                            <td>{props.info.snort_managers_info.ips[index]}</td>
                                            <td>{props.info.snort_managers_info.ports[index]}</td>
                                            {activeStatus(props.info.snort_managers_info.running)}
                                            <td>
                                            </td>
                                        </tr>
                                    )}
                                    {props.info.snort_managers_info.snort_statuses.map((status, index) =>
                                        <tr key={"snort_manager_status-" + index}>
                                            <td>Snort IDS</td>
                                            <td>{props.info.snort_managers_info.ips[index]}</td>
                                            <td></td>
                                            {activeStatus(status.running)}
                                            <td>
                                                <SpinnerOrButton
                                                    loading={loadingEntities.includes("snort_ids_manager")}
                                                    running={status.running}
                                                    entity={"snort_ids_manager"}/>
                                            </td>
                                        </tr>
                                    )}
                                    </tbody>
                                </Table>
                            </div>
                        </div>
                    </Collapse>
                </Card>


                <Card className="subCard">
                    <Card.Header>
                        <Button
                            onClick={() => setElkManagersOpen(!elkManagersOpen)}
                            aria-controls="elkManagersBody "
                            aria-expanded={elkManagersOpen}
                            variant="link"
                        >
                            <h5 className="semiTitle"> ELK Managers</h5>
                        </Button>
                    </Card.Header>
                    <Collapse in={elkManagersOpen}>
                        <div id="elkManagersBody" className="cardBodyHidden">
                            <div className="table-responsive">
                                <Table striped bordered hover>
                                    <thead>
                                    <tr>
                                        <th>Service</th>
                                        <th>IP</th>
                                        <th>Port</th>
                                        <th>Status</th>
                                        <th>Actions</th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    {props.info.elk_managers_info.elk_managers_statuses.map((status, index) =>
                                        <tr key={"elk_manager_status-" + index}>
                                            <td>ELK manager</td>
                                            <td>{props.info.elk_managers_info.ips[index]}</td>
                                            <td>{props.info.elk_managers_info.ports[index]}</td>
                                            {activeStatus(props.info.elk_managers_info.running)}
                                            <td>
                                            </td>
                                        </tr>
                                    )}
                                    {props.info.elk_managers_info.elk_managers_statuses.map((status, index) =>
                                        <tr key={"elk_manager_status-" + index}>
                                            <td>ELK stack</td>
                                            <td>{props.info.elk_managers_info.ips[index]}</td>
                                            <td>{props.execution.emulation_env_config.elk_config.elastic_port},
                                                {props.execution.emulation_env_config.elk_config.logstash_port},
                                                {props.execution.emulation_env_config.elk_config.kibana_port}
                                            </td>
                                            {activeStatus(props.info.elk_managers_info.running)}
                                            <td>
                                                <SpinnerOrButton
                                                    loading={loadingEntities.includes("elk_manager")}
                                                    running={props.info.elk_managers_info.running}
                                                    entity={"elk_manager"}/>
                                            </td>
                                        </tr>
                                    )}
                                    {props.info.elk_managers_info.elk_managers_statuses.map((status, index) =>
                                        <tr key={"elk_manager_status-" + index}>
                                            <td>Elasticsearch</td>
                                            <td>{props.info.elk_managers_info.ips[index]}</td>
                                            <td>{props.execution.emulation_env_config.elk_config.elastic_port}</td>
                                            {activeStatus(status.elasticRunning)}
                                            <td>
                                                <SpinnerOrButton
                                                    loading={loadingEntities.includes("elastic")}
                                                    running={status.elasticRunning}
                                                    entity={"elastic"}/>
                                            </td>
                                        </tr>
                                    )}
                                    {props.info.elk_managers_info.elk_managers_statuses.map((status, index) =>
                                        <tr key={"elk_manager_status-" + index}>
                                            <td>Logstash</td>
                                            <td>{props.info.elk_managers_info.ips[index]}</td>
                                            <td>{props.execution.emulation_env_config.elk_config.logstash_port}</td>
                                            {activeStatus(status.logstashRunning)}
                                            <td>
                                                <SpinnerOrButton
                                                    loading={loadingEntities.includes("logstash")}
                                                    running={status.logstashRunning}
                                                    entity={"logstash"}/>
                                            </td>
                                        </tr>
                                    )}

                                    {props.info.elk_managers_info.elk_managers_statuses.map((status, index) =>
                                        <tr key={"elk_manager_status-" + index}>
                                            <td>Kibana</td>
                                            <td>{props.info.elk_managers_info.ips[index]}</td>
                                            <td>{props.execution.emulation_env_config.elk_config.kibana_port}</td>
                                            {activeStatus(status.kibanaRunning)}
                                            <td>
                                                <SpinnerOrButton
                                                    loading={loadingEntities.includes("kibana")}
                                                    running={status.kibanaRunning}
                                                    entity={"kibana"}/>
                                            </td>
                                        </tr>
                                    )}
                                    </tbody>
                                </Table>
                            </div>
                        </div>
                    </Collapse>
                </Card>

                <Card className="subCard">
                    <Card.Header>
                        <Button
                            onClick={() => setTrafficManagersOpen(!trafficManagersOpen)}
                            aria-controls="trafficManagersBody"
                            aria-expanded={trafficManagersOpen}
                            variant="link"
                        >
                            <h5 className="semiTitle"> Traffic managers </h5>
                        </Button>
                    </Card.Header>
                    <Collapse in={trafficManagersOpen}>
                        <div id="trafficManagersBody" className="cardBodyHidden">
                            <div className="table-responsive">
                                <Table striped bordered hover>
                                    <thead>
                                    <tr>
                                        <th>Service</th>
                                        <th>IP</th>
                                        <th>Port</th>
                                        <th>Status</th>
                                        <th>Actions</th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    {props.info.traffic_managers_info.traffic_managers_statuses.map((status, index) =>
                                        <tr key={"traffic_manager_status-" + index}>
                                            <td>Traffic Manager</td>
                                            <td>{props.info.traffic_managers_info.ips[index]}</td>
                                            <td>{props.info.traffic_managers_info.ports[index]}</td>
                                            {activeStatus(props.info.traffic_managers_info.running)}
                                            <td>
                                            </td>
                                        </tr>
                                    )}
                                    {props.info.traffic_managers_info.traffic_managers_statuses.map((status, index) =>
                                        <tr key={"traffic_generator_status-" + index}>
                                            <td>Traffic Generator</td>
                                            <td>{props.info.traffic_managers_info.ips[index]}</td>
                                            <td>-</td>
                                            {activeStatus(status.running)}
                                            <td>
                                                <SpinnerOrButton
                                                    loading={loadingEntities.includes("traffic_manager")}
                                                    running={status.running}
                                                    entity={"traffic_manager"}/>
                                            </td>
                                        </tr>
                                    )}
                                    </tbody>
                                </Table>
                            </div>
                        </div>
                    </Collapse>
                </Card>

            </Card.Body>
        </Accordion.Collapse>
    </Card>)
}

ExecutionControlPlane.propTypes = {};
ExecutionControlPlane.defaultProps = {};
export default ExecutionControlPlane;
