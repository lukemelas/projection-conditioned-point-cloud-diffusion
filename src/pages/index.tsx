import { Link as ChakraLink, Text, Code, ListItem, Heading, UnorderedList } from '@chakra-ui/react'
import { Title, Authors } from 'components/Header'
import { Container } from 'components/Container'
import { DarkModeSwitch } from 'components/DarkModeSwitch'
import { LinksRow } from 'components/LinksRow'
import { Footer } from 'components/Footer'

import { title, abstract, citationId, citationAuthors, citationYear, citationBooktitle, acknowledgements, video_url } from 'data'


const Index = () => (
  <Container>

    {/* Heading */}
    <Title />
    <Authors />

    {/* Links */}
    <LinksRow />

    {/* Video */}
    <Container w="90vw" h="50.6vw" maxW="50rem" maxH="25rem" mb="3rem">
      <iframe
        width="100%" height="100%"
        src={video_url}
        title="Video"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen>
      </iframe>
    </Container>

    {/* Main */}
    <Container w="100%" maxW="60rem" alignItems="left" pl="1rem" pr="1rem">

      {/* Abstract */}
      <Heading fontWeight="light" fontSize="2xl" pb="1rem">Abstract</Heading>
      <Text pb="2rem">{abstract}</Text>

      {/* Example */}
      <Heading fontWeight="light" fontSize="2xl" pb="1rem">Examples</Heading>
      <img src={`${process.env.BASE_PATH || ""}/images/method-diagram-v3.png`} />
      <Text margin="auto" pt="0.5rem" pb="0.5rem" fontSize="small" paddingBottom="2rem">
      <Text fontWeight="bold" as="span">Method Diagram.</Text> <Text fontWeight="bold" as="span">PC<sup>2</sup></Text> reconstructs a colored point cloud from a single input image along with its camera pose.
        The method contains two sub-parts, both of which utilize our model projection conditioning method.
        First, we gradually denoise a set of points into the shape of an object.
        At each step in the diffusion process, we project image features onto the partially-denoised point cloud from the given camera pose, augmenting each point with a set of neural features.
        This step makes the diffusion process conditional on the image in a geometrically-consistent manner, enabling high-quality shape reconstruction.
        Second, we predict the color of each point using a model based on the same projection procedure.
        Hi
      </Text>

      {/* Example */}
      <img src={`${process.env.BASE_PATH || ""}/images/splash-figure.png`} />
      <Text margin="auto" pt="0.5rem" pb="0.5rem" fontSize="small" paddingBottom="2rem">
        <Text fontWeight="bold" as="span">PC<sup>2</sup></Text> performs single-image 3D point cloud reconstruction by gradually diffusing an initially random point cloud to align the with input image. It has been trained through simple, sparse COLMAP supervision from videos.
      </Text>
      <img src={`${process.env.BASE_PATH || ""}/images/results-figure-v3.png`} />
      <Text margin="auto" pt="0.5rem" pb="0.5rem" fontSize="small">
        Examples on three real-world categories from Co3D: toytrucks, teddy bears, and hydrants.
      </Text>

      {/* Citation */}
      <Heading fontWeight="light" fontSize="2xl" pt="2rem" pb="1rem">Citation</Heading>
      <Code p="0.5rem" borderRadius="5px" overflow="scroll" whiteSpace="nowrap">  {/*  fontFamily="monospace" */}
        @inproceedings&#123; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;{citationId}, <br />
          &nbsp;&nbsp;&nbsp;&nbsp;title=&#123;{title}&#125; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;author=&#123;{citationAuthors}&#125; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;year=&#123;{citationYear}&#125; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;booktitle=&#123;{citationBooktitle}&#125; <br />
        &#125;
      </Code>

      {/* Acknowledgements */}
      <Heading fontWeight="light" fontSize="2xl" pt="2rem" pb="1rem">Acknowledgements</Heading>
      <Text >
        {acknowledgements}
      </Text>
    </Container>

    <DarkModeSwitch />
    <Footer>
      <Text></Text>
    </Footer>
  </Container >
)

export default Index
